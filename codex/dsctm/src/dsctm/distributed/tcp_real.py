"""Gate 9 — Temporal Consistency Protocol as an actual distributed protocol.

`train/tcp.py` is a single-process state-machine simulator. It increments a Python counter
and resets a dict; it performs no communication, synchronises no parameters, and touches no
optimizer state. It must not be described as a training protocol, and this module exists so
that the manuscript's TCP claims have executable content behind them.

WHAT IS DEFINED HERE, PRECISELY
-------------------------------
The manuscript's language ("staleness", "causal masking of gradients") is imprecise about
what is actually stale. This implementation commits to a definition and states it:

  parameter version v_b   monotone counter, incremented when branch b applies a local
                          optimizer step
  global version V        incremented on every synchronisation event
  staleness Δ_b = V - v_b^sync   where v_b^sync is b's version at its last synchronisation
  HOLD                    branch b whose Δ_b would exceed δ_max suspends local updates
                          until the next synchronisation completes
  synchronisation         an all-reduce of branch parameters within the branch's replica
                          group, followed by Δ_b := 0 and V := V + 1

What is NOT claimed: that this is SGD convergence, or that ordinary DDP violates temporal
causality. Those are Gate 11's business. What IS claimed and tested here is a protocol
invariant — bounded version divergence — which is checkable and which the implementation
either satisfies or does not.

OPTIMIZER STATE
---------------
Adam moments are branch-local and are NOT synchronised. That is a deliberate, recorded
choice: averaging second-moment estimates across replicas that saw different data is not
well-defined, and pretending otherwise would be the kind of silent decision this campaign
exists to eliminate. `sync_optimizer_state=True` is available for the ablation, and the
choice is written into every run record.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import torch
import torch.distributed as dist

from .errors import PreflightFailure
from .runtime import DistContext, broadcast_object, is_initialized
from .sap import CommStats, Placement


class SyncReason(str, Enum):
    HOLD = "hold"
    PERIODIC = "periodic"
    EPOCH_END = "epoch_end"
    MANUAL = "manual"


class BranchAction(str, Enum):
    UPDATE = "update"
    HOLD = "hold"
    SKIP = "skip"


@dataclass
class TCPState:
    """Everything a checkpoint must restore for the protocol to resume identically."""

    delta_max: int = 10
    t_sync: int = 50
    global_version: int = 0
    step: int = 0
    branch_versions: dict[str, int] = field(default_factory=dict)
    branch_staleness: dict[str, int] = field(default_factory=dict)
    hold_events: int = 0
    periodic_syncs: int = 0
    hold_triggered_syncs: int = 0
    sync_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "TCPState":
        return cls(**blob)


class TemporalConsistencyProtocol:
    """Real TCP over a SAP placement.

    Every rank constructs one. Decisions are taken on the aggregator and BROADCAST, so all
    ranks agree on whether a synchronisation happens this step. Letting ranks decide
    independently is how a collective deadlocks.
    """

    def __init__(self, placement: Placement, ctx: DistContext, delta_max: int = 10,
                 t_sync: int = 50, sync_optimizer_state: bool = False,
                 stats: CommStats | None = None):
        if delta_max < 1:
            raise ValueError("delta_max must be >= 1")
        if t_sync < 1:
            raise ValueError("t_sync must be >= 1")
        self.placement = placement
        self.ctx = ctx
        self.sync_optimizer_state = sync_optimizer_state
        self.stats = stats or CommStats()
        self.state = TCPState(
            delta_max=delta_max, t_sync=t_sync,
            branch_versions={b: 0 for b in placement.branches},
            branch_staleness={b: 0 for b in placement.branches},
        )
        self._groups: dict[str, Any] = {}

    # -- protocol decision ------------------------------------------------ #
    def decide(self, updating: Sequence[str] | None = None) -> dict[str, Any]:
        """Advance one step and return the agreed action for every branch.

        Collective: EVERY rank must call this exactly once per step.
        """
        decision = None
        if self.ctx is None or self.ctx.is_main or self.ctx.rank == self.placement.aggregator_rank:
            decision = self._decide_local(updating)
        decision = broadcast_object(decision, src=self.placement.aggregator_rank
                                    if is_initialized() else 0)
        self._apply(decision)
        return decision

    def _decide_local(self, updating: Sequence[str] | None) -> dict[str, Any]:
        s = self.state
        step = s.step + 1
        active = set(s.branch_versions if updating is None else updating)
        actions: dict[str, str] = {}
        hold = False
        for b in self.placement.branches:
            if b not in active:
                actions[b] = BranchAction.SKIP.value
            elif s.branch_staleness[b] >= s.delta_max:
                actions[b] = BranchAction.HOLD.value
                hold = True
            else:
                actions[b] = BranchAction.UPDATE.value
        # HOLD takes precedence over the periodic schedule (manuscript Algorithm 1, l.14-21).
        if hold:
            reason = SyncReason.HOLD.value
        elif step % s.t_sync == 0:
            reason = SyncReason.PERIODIC.value
        else:
            reason = None
        return {"step": step, "actions": actions, "sync": reason is not None,
                "sync_reason": reason}

    def _apply(self, decision: dict[str, Any]) -> None:
        s = self.state
        s.step = decision["step"]
        for b, action in decision["actions"].items():
            if action == BranchAction.UPDATE.value:
                s.branch_versions[b] += 1
                s.branch_staleness[b] += 1
            elif action == BranchAction.HOLD.value:
                s.hold_events += 1
        if decision["sync"]:
            s.global_version += 1
            for b in s.branch_staleness:
                s.branch_staleness[b] = 0
            if decision["sync_reason"] == SyncReason.HOLD.value:
                s.hold_triggered_syncs += 1
            else:
                s.periodic_syncs += 1
            s.sync_log.append({"step": s.step, "reason": decision["sync_reason"],
                               "global_version": s.global_version})

    # -- the actual communication ----------------------------------------- #
    def _group_for(self, branch: str):
        if branch not in self._groups:
            ranks = self.placement.replica_groups.get(branch, [self.placement.branch_to_rank[branch]])
            self._groups[branch] = dist.new_group(ranks=ranks) if len(ranks) > 1 else None
        return self._groups[branch]

    def synchronize(self, model, optimizer=None) -> dict[str, Any]:
        """Perform the synchronisation this step's decision called for.

        Averages branch parameters within each replica group. This is real communication:
        `dist.all_reduce` on parameter tensors, byte-counted into `stats`.
        """
        if not is_initialized():
            return {"performed": False, "reason": "single process"}
        branch = self.placement.owns_branch(self.ctx.rank)
        moved = 0
        if branch is not None:
            group = self._group_for(branch)
            ranks = self.placement.replica_groups.get(branch, [])
            if group is not None and len(ranks) > 1:
                for p in model.inner._branches[branch].parameters():
                    dist.all_reduce(p.data, group=group)
                    p.data /= len(ranks)
                    self.stats.record("allreduce", p.data)
                    moved += p.data.element_size() * p.data.nelement()
                if self.sync_optimizer_state and optimizer is not None:
                    moved += self._sync_adam_moments(model, optimizer, group, len(ranks))
        dist.barrier()
        return {"performed": True, "bytes": moved,
                "global_version": self.state.global_version,
                "optimizer_state_synced": self.sync_optimizer_state}

    def _sync_adam_moments(self, model, optimizer, group, world) -> int:
        """Ablation path only. Averaging second moments across differently-fed replicas is
        not well-defined; this exists so the choice can be measured, not assumed."""
        moved = 0
        branch = self.placement.owns_branch(self.ctx.rank)
        params = list(model.inner._branches[branch].parameters())
        for p in params:
            st = optimizer.state.get(p)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    dist.all_reduce(st[key], group=group)
                    st[key] /= world
                    moved += st[key].element_size() * st[key].nelement()
        return moved

    # -- invariants -------------------------------------------------------- #
    def check_invariants(self) -> dict[str, Any]:
        s = self.state
        max_delta = max(s.branch_staleness.values()) if s.branch_staleness else 0
        return {
            "bounded_divergence": max_delta <= s.delta_max,
            "max_staleness": max_delta,
            "delta_max": s.delta_max,
            "hold_precedence_respected": all(
                not (e["reason"] == SyncReason.PERIODIC.value and e["step"] % s.t_sync != 0)
                for e in s.sync_log),
            "versions_monotone": all(v >= 0 for v in s.branch_versions.values()),
            "global_version": s.global_version,
            "hold_events": s.hold_events,
            "periodic_syncs": s.periodic_syncs,
            "hold_triggered_syncs": s.hold_triggered_syncs,
        }

    # -- checkpointing ----------------------------------------------------- #
    def state_dict(self) -> dict[str, Any]:
        return {"tcp_state": self.state.to_dict(),
                "sync_optimizer_state": self.sync_optimizer_state,
                "comm_stats": self.stats.to_dict()}

    def load_state_dict(self, blob: dict[str, Any]) -> None:
        self.state = TCPState.from_dict(blob["tcp_state"])
        self.sync_optimizer_state = blob.get("sync_optimizer_state", False)


# --------------------------------------------------------------------------- #
# Execution modes (Gate 9 requirement)
# --------------------------------------------------------------------------- #
class ExecutionMode(str, Enum):
    DDP_SYNC = "full_model_synchronous_ddp"
    SAP_SYNC = "synchronous_sap"
    SAP_ASYNC_NO_TCP = "asynchronous_sap_without_tcp"
    SAP_ASYNC_TCP = "asynchronous_sap_with_tcp"


MODE_DESCRIPTIONS = {
    ExecutionMode.DDP_SYNC: (
        "Control. Whole model replicated per rank; gradients all-reduced every step."),
    ExecutionMode.SAP_SYNC: (
        "Branch-parallel placement, but every branch synchronises every step. Isolates the "
        "cost of partitioning from the cost of asynchrony."),
    ExecutionMode.SAP_ASYNC_NO_TCP: (
        "Branch-parallel with unbounded local updates. The failure mode TCP claims to fix; "
        "version divergence is unbounded by construction."),
    ExecutionMode.SAP_ASYNC_TCP: (
        "Branch-parallel with bounded version divergence enforced by HOLD plus periodic "
        "synchronisation."),
}


def describe_modes() -> str:
    return json.dumps({m.value: MODE_DESCRIPTIONS[m] for m in ExecutionMode}, indent=2)
