"""Temporal Coordination Protocol (TCP) — single-process simulation of Algorithm 1.

The manuscript's TCP is a distributed training protocol. Its *invariants* — staleness
increment/reset, HOLD activation, HOLD-vs-periodic precedence, and the δ ≤ δ_max
bound — do not require a real multi-GPU cluster to verify. This module simulates the
version-lag bookkeeping in one process so EXP-0.3 can test those invariants directly
(the multi-GPU *performance* claims still need the 8-GPU server; that is GAP-5).

Staleness is defined as a parameter-version lag  Δ_b = v_global − v_b.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def causal_gradient_mask(grad_time, t_current: int):
    """Eq. 12: zero gradient contributions from future timesteps (t > t_current).

    grad_time: tensor with time on the last axis (..., T). Returns a masked copy.
    """
    import torch

    T = grad_time.shape[-1]
    mask = (torch.arange(T, device=grad_time.device) <= t_current).to(grad_time.dtype)
    return grad_time * mask


@dataclass
class StalenessController:
    branches: tuple = ("ssb", "msb", "lsb")
    delta_max: int = 10
    t_sync: int = 50
    delta: dict = field(default_factory=dict)
    global_version: int = 0
    step_count: int = 0
    events: list = field(default_factory=list)

    def __post_init__(self):
        if not self.delta:
            self.delta = {b: 0 for b in self.branches}

    def _allreduce(self, reason: str):
        self.global_version += 1
        for b in self.branches:
            self.delta[b] = 0
        self.events.append((self.step_count, "allreduce", reason))

    def step(self, updating_branches: Optional[list] = None) -> dict:
        """Advance one training step. Per-branch action:
        'update' (local update applied, Δ+=1), 'hold' (Δ would exceed δ_max →
        suspend + AllReduce + reset), or 'skip' (branch not updating this step)."""
        self.step_count += 1
        updating = set(self.branches if updating_branches is None else updating_branches)
        actions, held = {}, False
        for b in self.branches:
            if b not in updating:
                actions[b] = "skip"
            elif self.delta[b] >= self.delta_max:
                actions[b], held = "hold", True
            else:
                self.delta[b] += 1
                actions[b] = "update"
        if held:                                   # HOLD takes precedence
            self._allreduce("hold")
        elif self.step_count % self.t_sync == 0:   # else periodic AllReduce
            self._allreduce("periodic")
        return actions
