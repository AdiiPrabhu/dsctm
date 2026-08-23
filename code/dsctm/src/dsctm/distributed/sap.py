"""Gate 8 — Scale-Aware Partitioner: genuine branch-parallel execution.

Full-model DDP replicates the whole network on every rank. SAP does something different:
each temporal branch lives on its own rank, activations are shipped to an aggregator rank
that owns CSAG + FiLM + head, and gradients flow back along the same edges.

This is NOT DDP with extra steps, and the distinction is the whole point of the
manuscript's §III-C. Full-model DDP remains the CONTROL (DECISIONS.md D-007); no SAP claim
is meaningful without it.

Topology. A PARAM node has 2 V100s, so the manuscript's Fig. 3 layout (3 branches + 1
aggregator) needs >= 4 ranks = 2 nodes:

    rank 0  SSB          rank 2  LSB
    rank 1  MSB          rank 3  aggregator (CSAG + FiLM + head + loss)

Autograd across ranks is implemented with a matched pair of autograd Functions: the send
side's backward is a receive, and the receive side's backward is a send. Shapes are
exchanged once at setup so no metadata travels on the hot path.

Correctness is defined by equivalence: SAP forward and backward must match the monolithic
single-process model to within a declared tolerance. `tests/test_sap.py` asserts exactly
that before any performance number is taken seriously.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from .errors import PreflightFailure
from .runtime import DistContext, is_initialized

BRANCH_ORDER = ("ssb", "msb", "lsb")   # deterministic; never iterate a set here


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Placement:
    """Which rank owns what. Deterministic given (branches, world_size)."""

    branch_to_rank: dict[str, int]
    aggregator_rank: int
    world_size: int
    replica_groups: dict[str, list[int]] = field(default_factory=dict)

    @property
    def branches(self) -> tuple[str, ...]:
        return tuple(b for b in BRANCH_ORDER if b in self.branch_to_rank)

    def role(self, rank: int) -> str:
        if rank == self.aggregator_rank:
            return "aggregator"
        for b, r in self.branch_to_rank.items():
            if r == rank:
                return f"branch:{b}"
        return "idle"

    def owns_branch(self, rank: int) -> str | None:
        for b in self.branches:
            if self.branch_to_rank[b] == rank:
                return b
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"branch_to_rank": dict(self.branch_to_rank),
                "aggregator_rank": self.aggregator_rank,
                "world_size": self.world_size,
                "replica_groups": {k: list(v) for k, v in self.replica_groups.items()},
                "roles": {r: self.role(r) for r in range(self.world_size)}}


def plan_placement(branches: Sequence[str], world_size: int,
                   compute_cost: dict[str, float] | None = None,
                   comm_cost: dict[str, float] | None = None) -> Placement:
    """Assign branches to ranks.

    Implements manuscript Eq. (11): the load score
    ``L_b = C_b^compute / (C_b^compute + C_b^comm)`` orders branches, and they are placed
    onto the least-loaded ranks in descending order. With one rank per branch plus a
    dedicated aggregator this reduces to a fixed assignment, but the score is computed and
    recorded so the ordering is auditable rather than incidental.

    For ``world_size`` beyond ``len(branches) + 1`` the surplus ranks become data-parallel
    replicas of the heaviest branches (manuscript §III-C, N > 3), recorded in
    ``replica_groups``.
    """
    branches = [b for b in BRANCH_ORDER if b in branches]
    if not branches:
        raise ValueError("no branches to place")
    need = len(branches) + 1
    if world_size < need:
        raise PreflightFailure(
            f"SAP needs at least {need} ranks for {len(branches)} branches plus an "
            f"aggregator, but world_size={world_size}. On PARAM a node has 2 GPUs, so "
            f"the manuscript's 3-branch layout requires --nodes=2 --gres=gpu:2.")

    # Eq. 11 load score. Defaults reflect the measured cost profile: dense small dilations
    # (SSB) are compute-heavy, sparse large dilations (LSB) are comparatively light.
    compute_cost = compute_cost or {"ssb": 0.87, "msb": 0.72, "lsb": 0.54}
    comm_cost = comm_cost or {b: 0.20 for b in branches}
    scores = {b: compute_cost.get(b, 0.5) /
                 (compute_cost.get(b, 0.5) + comm_cost.get(b, 0.2)) for b in branches}
    ordered = sorted(branches, key=lambda b: (-scores[b], b))

    aggregator_rank = world_size - 1
    worker_ranks = [r for r in range(world_size) if r != aggregator_rank]
    branch_to_rank = {b: worker_ranks[i] for i, b in enumerate(ordered)}

    replica_groups: dict[str, list[int]] = {b: [branch_to_rank[b]] for b in ordered}
    for i, spare in enumerate(worker_ranks[len(ordered):]):
        target = ordered[i % len(ordered)]           # heaviest branches replicate first
        replica_groups[target].append(spare)

    return Placement(branch_to_rank=branch_to_rank, aggregator_rank=aggregator_rank,
                     world_size=world_size, replica_groups=replica_groups)


# --------------------------------------------------------------------------- #
# Autograd-aware point-to-point transfer
# --------------------------------------------------------------------------- #
class _SendActivation(torch.autograd.Function):
    """Forward: send tensor to ``dst``. Backward: receive its gradient from ``dst``."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dst: int) -> torch.Tensor:
        ctx.dst = dst
        ctx.shape = tuple(tensor.shape)
        ctx.dtype = tensor.dtype
        ctx.device = tensor.device
        dist.send(tensor.detach().contiguous(), dst=dst)
        # Return a token so the graph has an output; it carries no value.
        return tensor.new_zeros(1)

    @staticmethod
    def backward(ctx, _grad_token):
        grad = torch.empty(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        dist.recv(grad, src=ctx.dst)
        return grad, None


class _RecvActivation(torch.autograd.Function):
    """Forward: receive from ``src``. Backward: send the gradient back to ``src``."""

    @staticmethod
    def forward(ctx, template: torch.Tensor, src: int) -> torch.Tensor:
        ctx.src = src
        buf = torch.empty_like(template)
        dist.recv(buf, src=src)
        return buf

    @staticmethod
    def backward(ctx, grad_output):
        dist.send(grad_output.detach().contiguous(), dst=ctx.src)
        return None, None


def send_activation(tensor: torch.Tensor, dst: int) -> torch.Tensor:
    return _SendActivation.apply(tensor, dst)


def recv_activation(template: torch.Tensor, src: int) -> torch.Tensor:
    return _RecvActivation.apply(template, src)


# --------------------------------------------------------------------------- #
# Communication accounting  (tracker E4-17 — measured, not estimated)
# --------------------------------------------------------------------------- #
@dataclass
class CommStats:
    forward_bytes: int = 0
    backward_bytes: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    allreduce_bytes: int = 0
    allreduce_calls: int = 0
    wall_seconds: float = 0.0

    def record(self, kind: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.element_size() * tensor.nelement()
        if kind == "forward":
            self.forward_bytes += nbytes
            self.forward_calls += 1
        elif kind == "backward":
            self.backward_bytes += nbytes
            self.backward_calls += 1
        else:
            self.allreduce_bytes += nbytes
            self.allreduce_calls += 1

    @property
    def total_bytes(self) -> int:
        return self.forward_bytes + self.backward_bytes + self.allreduce_bytes

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "total_bytes": self.total_bytes,
                "total_mib": round(self.total_bytes / 2**20, 4)}


def predicted_bytes_per_sample(T: int, D: int, n_branches: int,
                               dtype_bytes: int = 4) -> dict[str, int]:
    """The manuscript's §III-F analytical prediction, kept beside the measurement.

    Forward: each branch ships H_b of shape (T, D). Backward: the same shape returns.
    Reporting both makes the E4-17 comparison ('validate communication-volume
    calculations with instrumentation') a checkable claim rather than an assertion.
    """
    per_branch = T * D * dtype_bytes
    return {"forward": per_branch * n_branches,
            "backward": per_branch * n_branches,
            "total": 2 * per_branch * n_branches}


# --------------------------------------------------------------------------- #
# The partitioned model
# --------------------------------------------------------------------------- #
class SAPModel(nn.Module):
    """Branch-parallel D-MSTCN.

    Every rank builds the full module set so parameter shapes and initial values agree, but
    each rank only *executes* its assigned part. Unused parameters are frozen so no
    optimizer touches them and no gradient is expected for them.
    """

    def __init__(self, model: nn.Module, placement: Placement, ctx: DistContext,
                 stats: CommStats | None = None):
        super().__init__()
        self.inner = model
        self.placement = placement
        self.ctx = ctx
        self.stats = stats or CommStats()
        self.role = placement.role(ctx.rank)
        self.my_branch = placement.owns_branch(ctx.rank)
        self.is_aggregator = ctx.rank == placement.aggregator_rank
        self._freeze_unused()

    def _freeze_unused(self) -> None:
        """Only the parts this rank executes may carry gradients."""
        for name, module in self.inner._branches.items():
            active = (name == self.my_branch)
            for p in module.parameters():
                p.requires_grad_(active)
        for module in (self.inner.csag, self.inner.film, self.inner.head):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(self.is_aggregator)
        # The input projection runs on every branch rank AND the aggregator needs it for
        # nothing, so it is owned by branch ranks only.
        for p in self.inner.input_proj.parameters():
            p.requires_grad_(self.my_branch is not None)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Aggregator returns logits; branch ranks return None. Every rank must call this.

        Communication is deliberately kept OUTSIDE the autograd graph. The obvious design —
        a pair of autograd Functions whose backward issues `dist.send`/`dist.recv` — fails
        in practice: PyTorch runs backward on a dedicated worker thread, and gloo binds
        transport buffers per thread, so p2p from the autograd thread aborts the process
        with `gloo::EnforceNotMet: Cannot lock pointer to unbound buffer`. Disabling
        autograd multithreading does not reliably fix it either.

        So the graph is cut at the rank boundary and re-joined manually:

          forward   branch computes H_b (graph retained locally) and sends H_b.detach()
                    aggregator receives into a LEAF tensor with requires_grad=True
          backward  aggregator runs loss.backward(), reads leaf.grad, sends it back
                    branch receives that grad and runs H_b.backward(grad)

        Every collective therefore executes on the main thread, where the process group was
        created. This is the standard manual model-parallel pattern and it is what makes
        the equivalence tests pass rather than crash.
        """
        B, T, _ = X.shape
        D = self.inner.cfg.D
        self._received = {}
        self._local_out = None

        if self.my_branch is not None:
            h = self.inner.input_proj(X).transpose(1, 2)
            out = self.inner._branches[self.my_branch](h).transpose(1, 2)
            self._local_out = out                      # graph kept on this rank
            payload = out.detach().contiguous()
            self.stats.record("forward", payload)
            dist.send(payload, dst=self.placement.aggregator_rank)
            return None

        if self.is_aggregator:
            branch_outs = []
            for b in self.placement.branches:
                src = self.placement.branch_to_rank[b]
                buf = torch.empty(B, T, D, dtype=X.dtype, device=X.device)
                dist.recv(buf, src=src)
                self.stats.record("forward", buf)
                buf.requires_grad_(True)               # leaf: its .grad is what we ship back
                self._received[b] = buf
                branch_outs.append(buf)
            if len(branch_outs) == 1:
                fused = branch_outs[0]
            elif self.inner.cfg.csag_mode == "mean":
                fused = torch.stack(branch_outs, 0).mean(0)
            else:
                fused, _ = self.inner.csag(branch_outs)
            Hp = self.inner.film(fused, subject_idx) if self.inner.film is not None else fused
            return self.inner.head(Hp, mask)
        return None

    def backward_from(self, loss: torch.Tensor | None) -> None:
        """Complete backward across the rank boundary. Every rank must call this."""
        if self.is_aggregator:
            if loss is None:
                raise PreflightFailure("aggregator must supply a loss")
            loss.backward()
            for b in self.placement.branches:                 # deterministic order
                buf = self._received[b]
                grad = (buf.grad if buf.grad is not None
                        else torch.zeros_like(buf)).contiguous()
                self.stats.record("backward", grad)
                dist.send(grad, dst=self.placement.branch_to_rank[b])
            self._received = {}
            return

        if self.my_branch is not None:
            if self._local_out is None:
                raise PreflightFailure("forward() must run before backward_from()")
            grad = torch.empty_like(self._local_out)
            dist.recv(grad, src=self.placement.aggregator_rank)
            self.stats.record("backward", grad)
            self._local_out.backward(grad)
            self._local_out = None


def sap_step(model: SAPModel, X, y, subject_idx, mask, loss_fn, optimizer) -> float | None:
    """One SAP training step. Every rank calls it; only the aggregator sees the loss."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(X, subject_idx, mask)
    loss = loss_fn(logits, y) if model.is_aggregator else None
    model.backward_from(loss)
    optimizer.step()
    return float(loss.detach()) if loss is not None else None


def replicate_gradients(model: SAPModel) -> None:
    """Average gradients within each branch's replica group (world_size > branches + 1)."""
    if not is_initialized():
        return
    groups = model.placement.replica_groups
    branch = model.my_branch
    if branch is None or len(groups.get(branch, [])) < 2:
        return
    ranks = groups[branch]
    group = dist.new_group(ranks=ranks)
    for p in model.inner._branches[branch].parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=group)
            p.grad /= len(ranks)
            model.stats.record("allreduce", p.grad)
    dist.destroy_process_group(group)
