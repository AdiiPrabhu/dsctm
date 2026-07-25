"""D-MSTCN top-level model (manuscript Fig. 1 pipeline):

    X → input projection (Eq.1) → {SSB, MSB, LSB} branches → CSAG fusion (Eqs.3-6)
      → FiLM subject adapter (Eqs.7-9) → mean-pool + MLP head (Eq.10) → logits
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .blocks import CSAG, Branch, FiLMAdapter, Head


@dataclass
class DMSTCNConfig:
    input_dim: int = 8          # StudentLife=8, DAIC-WOZ=88, SEED=310
    n_classes: int = 3          # StudentLife=3, DAIC=2, SEED=3
    D: int = 128                # shared width (Eq.1)
    K: int = 3
    n_subjects: int = 48
    d_s: int = 8
    film_hidden: int = 32       # NOT SPECIFIED in paper; documented default
    head_hidden: int = 128      # NOT SPECIFIED in paper; documented default
    dropout: float = 0.0        # NOT SPECIFIED in paper; default off
    temperature: float | None = None  # default √D
    ssb: tuple = (1, 2, 4, 8)
    msb: tuple = (8, 16, 32, 64)
    lsb: tuple = (32, 64, 128, 256)
    enabled_branches: tuple = ("ssb", "msb", "lsb")  # for branch ablations (EXP-5.1)
    use_film: bool = True       # for personalization ablation (EXP-5.5)
    film_mode: str = "subject"  # subject | global | global_matched
    csag_mode: str = "attention"  # attention | mean | static


class DMSTCN(nn.Module):
    def __init__(self, cfg: DMSTCNConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.D)  # Eq.1 (shared)
        self._branches = nn.ModuleDict()
        schedules = {"ssb": cfg.ssb, "msb": cfg.msb, "lsb": cfg.lsb}
        for name in cfg.enabled_branches:
            self._branches[name] = Branch(cfg.D, cfg.K, schedules[name], cfg.dropout)
        n_active = len(cfg.enabled_branches)
        if cfg.csag_mode not in ("attention", "mean", "static"):
            raise ValueError(f"unknown csag_mode {cfg.csag_mode!r}")
        self.csag = (CSAG(cfg.D, n_active, cfg.temperature)
                     if n_active > 1 and cfg.csag_mode == "attention" else None)
        self.static_alpha = (nn.Parameter(torch.zeros(n_active))
                             if n_active > 1 and cfg.csag_mode == "static" else None)
        if cfg.film_mode not in ("subject", "global", "global_matched"):
            raise ValueError(f"unknown film_mode {cfg.film_mode!r}")
        film_subjects = 1 if cfg.film_mode == "global" else cfg.n_subjects
        self.film = (
            FiLMAdapter(cfg.D, film_subjects, cfg.d_s, cfg.film_hidden)
            if cfg.use_film
            else None
        )
        self.head = Head(cfg.D, cfg.n_classes, cfg.head_hidden)

    def forward(self, X, subject_idx=None, mask=None, return_aux: bool = False):
        # X: (B, T, F)
        h = self.input_proj(X).transpose(1, 2)  # (B, D, T)
        branch_outs = [self._branches[n](h).transpose(1, 2) for n in self.cfg.enabled_branches]
        alpha = None
        if len(branch_outs) == 1:
            fused = branch_outs[0]
        elif self.cfg.csag_mode == "mean":
            fused = torch.stack(branch_outs, 0).mean(0)  # noCSAG: fixed average
        elif self.cfg.csag_mode == "static":
            alpha = torch.softmax(self.static_alpha, dim=0)
            fused = sum(alpha[i] * value for i, value in enumerate(branch_outs))
        else:
            fused, alpha = self.csag(branch_outs)
        if self.film is not None:
            if subject_idx is None:
                raise ValueError("FiLM-enabled D-MSTCN requires subject_idx")
            if self.cfg.film_mode in ("global", "global_matched"):
                subject_idx = torch.zeros_like(subject_idx)
            Hp = self.film(fused, subject_idx)
        else:
            Hp = fused
        logits = self.head(Hp, mask)
        if return_aux:
            return logits, {"alpha": alpha, "branches": branch_outs, "fused": fused}
        return logits

    def branch(self, name: str) -> Branch:
        return self._branches[name]
