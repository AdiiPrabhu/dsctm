"""D-MSTCN building blocks, implemented to the manuscript equations (pp. 7-8).

Conventions: tensors flow as (B, T, D) at module boundaries; convolutions operate
internally in (B, D, T). All temporal convolutions are strictly causal (left-padded),
so output t never depends on input > t — this is asserted by EXP-0.4.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Dilated causal 1-D convolution: left-pad by (K-1)*dilation, no future leak."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):  # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class DilatedResidualBlock(nn.Module):
    """Eq. 2:  H^ℓ = H^{ℓ-1} + Conv_r( GELU( Conv_r( LN(H^{ℓ-1}) ) ) ).

    Two causal convolutions per block at the same dilation r, LayerNorm at block
    input (over the D channel dim), GELU between the convs, identity residual add.
    """

    def __init__(self, D: int, K: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.conv1 = CausalConv1d(D, K, dilation)
        self.conv2 = CausalConv1d(D, K, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, D, T)
        h = self.ln(x.transpose(1, 2)).transpose(1, 2)
        h = self.conv1(h)
        h = F.gelu(h)
        h = self.conv2(h)
        h = self.drop(h)
        return x + h


class Branch(nn.Module):
    """One temporal branch: L_b dilated residual blocks over a dilation schedule."""

    def __init__(self, D: int, K: int, dilations, dropout: float = 0.0):
        super().__init__()
        self.dilations = list(dilations)
        self.K = K
        self.blocks = nn.ModuleList(
            [DilatedResidualBlock(D, K, r, dropout) for r in self.dilations]
        )

    def forward(self, x):  # x: (B, D, T)
        for blk in self.blocks:
            x = blk(x)
        return x

    def theoretical_rf_two_conv(self) -> int:
        """RF assuming two convs per block: R = 1 + 2(K-1)·Σ r_l."""
        return 1 + 2 * (self.K - 1) * sum(self.dilations)

    def theoretical_rf_one_conv(self) -> int:
        """RF assuming one conv per block: R = 1 + (K-1)·Σ r_l (for reference)."""
        return 1 + (self.K - 1) * sum(self.dilations)


class CSAG(nn.Module):
    """Cross-Scale Attention Gate, Eqs. 3-6.

        Z = W_z·[H_s;H_m;H_l] + b_z    (3)   3D -> 3D
        A = W_α·Z + b_α                (4)   3D -> n_branches
        α = softmax(A / √D)            (5)   per-timestep weights over branches
        H = Σ_b α_{:,b} ⊙ H_b          (6)
    """

    def __init__(self, D: int, n_branches: int = 3, temperature: float | None = None):
        super().__init__()
        self.D = D
        self.n = n_branches
        self.W_z = nn.Linear(n_branches * D, n_branches * D)
        self.W_a = nn.Linear(n_branches * D, n_branches)
        self.temperature = float(temperature) if temperature else math.sqrt(D)

    def forward(self, branch_outs):  # list of n × (B, T, D)
        H = torch.stack(branch_outs, dim=2)          # (B, T, n, D)
        cat = torch.cat(branch_outs, dim=-1)         # (B, T, nD)
        A = self.W_a(self.W_z(cat))                  # (B, T, n)
        alpha = torch.softmax(A / self.temperature, dim=-1)  # (B, T, n)
        out = (alpha.unsqueeze(-1) * H).sum(dim=2)   # (B, T, D)
        return out, alpha


class FiLMAdapter(nn.Module):
    """FiLM subject personalization, Eqs. 7-9.

    Per-subject learned embedding e_s ∈ R^{d_s} (the ONLY per-subject stored cost).
    A SHARED generator MLP maps e_s → per-channel (γ, β) ∈ R^D; H' = γ ⊙ H + β.
    (This separation is exactly the parameter-accounting fix flagged in EXP-0.2:
    per-subject storage is d_s, not 2D.)
    """

    def __init__(self, D: int, n_subjects: int, d_s: int = 8, hidden: int = 32):
        super().__init__()
        self.d_s = d_s
        self.embed = nn.Embedding(n_subjects, d_s)
        self.shared = nn.Linear(d_s, hidden)
        self.to_gamma = nn.Linear(hidden, D)
        self.to_beta = nn.Linear(hidden, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        # initialize FiLM as identity (γ≈1, β≈0) for stable training start
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, H, subject_idx, e_override=None):  # H: (B, T, D)
        e = self.embed(subject_idx) if e_override is None else e_override
        h = F.relu(self.shared(e))
        gamma = self.to_gamma(h).unsqueeze(1)  # (B, 1, D)
        beta = self.to_beta(h).unsqueeze(1)
        return gamma * H + beta


class Head(nn.Module):
    """Eq. 10: mean-pool over time → 2-layer MLP → logits (softmax in the loss)."""

    def __init__(self, D: int, n_classes: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or D
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, H, mask=None):  # H: (B, T, D); mask: (B, T), True=valid
        if mask is None:
            pooled = H.mean(dim=1)
        else:
            w = mask.to(dtype=H.dtype).unsqueeze(-1)
            pooled = (H * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return self.fc2(F.relu(self.fc1(pooled)))
