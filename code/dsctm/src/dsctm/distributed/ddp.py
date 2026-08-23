"""DDP model wrapping, lazy-parameter materialization, and coordinated early stopping.

Three specific hazards this module closes, all of which were identified in the Gate 0
audit as things that would break on PARAM:

1. ``ITransformerBaseline`` uses ``nn.LazyLinear``. DDP requires every parameter to exist
   at wrap time; wrapping a module with uninitialized lazy parameters raises. Fixed by a
   deterministic dry forward before the wrap.
2. Early stopping decided independently per rank. If rank 0 breaks the epoch loop and
   rank 1 does not, rank 1 blocks forever in the next gradient all-reduce and the job
   burns its remaining wall-clock. Fixed by deciding on rank 0 and broadcasting.
3. Model state diverging across ranks. DDP broadcasts at construction, but a later
   rank-local mutation (loading a checkpoint on one rank only, say) is invisible until the
   results are wrong. ``assert_replicas_agree`` makes it loud.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import state_digest
from .errors import PreflightFailure
from .runtime import DistContext, broadcast_object, is_initialized


def has_lazy_parameters(model: nn.Module) -> bool:
    """True if any parameter is still an uninitialized lazy placeholder."""
    for param in model.parameters():
        if isinstance(param, nn.parameter.UninitializedParameter):
            return True
    for buffer in model.buffers():
        if isinstance(buffer, nn.parameter.UninitializedBuffer):
            return True
    return False


def materialize_lazy_parameters(model: nn.Module,
                                dry_run: Callable[[nn.Module], Any] | None = None,
                                example_input: tuple | None = None) -> bool:
    """Force lazy modules to build their real parameters. Returns True if work was done.

    Supply either ``dry_run`` (a callable taking the model) or ``example_input`` (args
    tuple forwarded to ``model(*example_input)``). The pass runs under ``no_grad`` in eval
    mode and is discarded; it exists purely to fix parameter shapes.
    """
    if not has_lazy_parameters(model):
        return False
    if dry_run is None and example_input is None:
        raise PreflightFailure(
            "model has uninitialized lazy parameters and no dry-run input was supplied. "
            "DDP cannot wrap a module with UninitializedParameter. Pass example_input=..."
        )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if dry_run is not None:
            dry_run(model)
        else:
            model(*example_input)
    model.train(was_training)
    if has_lazy_parameters(model):
        raise PreflightFailure(
            "lazy parameters remain uninitialized after the dry-run pass; "
            "the example input did not exercise every lazy module"
        )
    return True


def wrap_ddp(model: nn.Module, ctx: DistContext, *,
             example_input: tuple | None = None,
             dry_run: Callable[[nn.Module], Any] | None = None,
             find_unused_parameters: bool = False,
             broadcast_buffers: bool = True,
             gradient_as_bucket_view: bool = True) -> nn.Module:
    """Move the model to this rank's device and wrap it in DDP.

    Returns the bare model unchanged when the job is single-process, so the same training
    code runs under ``python`` and under ``torchrun`` with no branching at the call site.

    ``find_unused_parameters`` defaults to False deliberately. It is a real performance
    cost and it masks genuine graph bugs. D-MSTCN branch ablations remove whole branches
    from the module, not from the graph, so every remaining parameter does receive a
    gradient. If a future variant genuinely has conditionally-unused parameters, set it
    explicitly and record why.
    """
    model = model.to(ctx.device)
    materialize_lazy_parameters(model, dry_run=dry_run, example_input=example_input) \
        if (example_input is not None or dry_run is not None) else None
    if has_lazy_parameters(model):
        raise PreflightFailure(
            "refusing to wrap a model with uninitialized lazy parameters "
            "(nn.LazyLinear); pass example_input= to wrap_ddp"
        )
    if not ctx.is_distributed or not is_initialized():
        return model
    kwargs: dict[str, Any] = {
        "find_unused_parameters": find_unused_parameters,
        "broadcast_buffers": broadcast_buffers,
        "gradient_as_bucket_view": gradient_as_bucket_view,
    }
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
        kwargs["output_device"] = ctx.local_rank
    return DistributedDataParallel(model, **kwargs)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def assert_replicas_agree(model: nn.Module, label: str = "model state") -> str:
    """Every rank must hold identical weights. Returns the shared digest."""
    digest = state_digest(model)
    if not is_initialized():
        return digest
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise PreflightFailure(
            f"{label} differs across ranks: {gathered}. Replicas have diverged; "
            f"any result from this job is invalid."
        )
    return digest


# --------------------------------------------------------------------------- #
# Coordinated early stopping
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopDecision:
    improved: bool
    should_stop: bool
    best_score: float
    patience: int
    epoch: int


class EarlyStopCoordinator:
    """Decide on rank 0, broadcast to all. Never let ranks decide independently.

    Usage::

        stopper = EarlyStopCoordinator(patience=15, ctx=ctx)
        for epoch in range(max_epochs):
            train_one_epoch(...)
            dev_score = evaluate(...)          # already reduced across ranks
            d = stopper.step(dev_score, epoch)
            if d.improved:
                save_checkpoint(...)           # rank 0 only, internally
            if d.should_stop:
                break                          # every rank breaks on the same epoch
    """

    def __init__(self, patience: int, ctx: DistContext | None = None,
                 mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = int(patience)
        self.ctx = ctx
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = -1

    def _is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, epoch: int) -> EarlyStopDecision:
        """Collective. EVERY rank must call this exactly once per epoch."""
        is_main = self.ctx is None or self.ctx.is_main
        if is_main:
            improved = self._is_better(score)
            if improved:
                self.best_score = float(score)
                self.counter = 0
                self.best_epoch = int(epoch)
            else:
                self.counter += 1
            decision = EarlyStopDecision(
                improved=improved,
                should_stop=self.counter >= self.patience,
                best_score=self.best_score,
                patience=self.counter,
                epoch=int(epoch),
            )
        else:
            decision = None

        decision = broadcast_object(decision, src=0)
        # Keep non-main ranks' local view consistent so a later restore/report agrees.
        self.best_score = decision.best_score
        self.counter = decision.patience
        if decision.improved:
            self.best_epoch = decision.epoch
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {"best_score": self.best_score, "counter": self.counter,
                "best_epoch": self.best_epoch, "patience": self.patience, "mode": self.mode}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_score = state["best_score"]
        self.counter = state["counter"]
        self.best_epoch = state["best_epoch"]


def build_grad_scaler(precision: str, device: torch.device):
    """GradScaler for fp16, disabled otherwise.

    fp16 on V100 without a scaler underflows small gradients to zero and the loss curve
    looks 'fine' while the model quietly learns less. The scaler is mandatory, not
    optional, at fp16.
    """
    enabled = precision.lower() in ("fp16", "float16", "half") and device.type == "cuda"
    try:  # torch >= 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)
