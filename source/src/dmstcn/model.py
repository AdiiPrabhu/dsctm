"""Single-device D-MSTCN architecture.

Inputs use the public `(batch, time, features)` convention. Convolutions are
internally channel-first and are explicitly left padded to guarantee causality.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dmstcn.config import DMSTCNConfig


def theoretical_receptive_field(kernel_size: int, dilations: tuple[int, ...], convolutions_per_block: int = 2) -> int:
    """Return the receptive field for stride-one causal convolutions."""
    if kernel_size <= 0 or convolutions_per_block <= 0 or any(d <= 0 for d in dilations):
        raise ValueError("kernel size, dilations, and convolution count must be positive")
    return 1 + convolutions_per_block * (kernel_size - 1) * sum(dilations)


class CausalConv1d(nn.Conv1d):
    """A Conv1d with left-only padding and length-preserving output."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__(channels, channels, kernel_size, dilation=dilation, padding=0)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return super().forward(F.pad(inputs, (self.left_padding, 0)))


class ResidualCausalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv1 = CausalConv1d(hidden_dim, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_dim, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = F.gelu(self.conv1(hidden))
        hidden = self.dropout(self.conv2(hidden)).transpose(1, 2)
        return residual + hidden


class TemporalBranch(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilations: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualCausalBlock(hidden_dim, kernel_size, dilation, dropout)
            for dilation in dilations
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class CrossScaleAttentionGate(nn.Module):
    def __init__(self, hidden_dim: int, num_branches: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches
        self.context = nn.Linear(hidden_dim * num_branches, hidden_dim * num_branches)
        self.scores = nn.Linear(hidden_dim * num_branches, num_branches)

    def forward(self, branches: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        if len(branches) != self.num_branches:
            raise ValueError(f"expected {self.num_branches} branches, got {len(branches)}")
        concatenated = torch.cat(branches, dim=-1)
        logits = self.scores(self.context(concatenated))
        weights = torch.softmax(logits / self.hidden_dim**0.5, dim=-1)
        stacked = torch.stack(branches, dim=-2)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return fused, weights


class SubjectFiLM(nn.Module):
    def __init__(self, num_subjects: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_subjects, embedding_dim)
        self.shared = nn.Linear(embedding_dim, hidden_dim)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: Tensor, subject_ids: Tensor) -> Tensor:
        conditioning = F.relu(self.shared(self.embedding(subject_ids)))
        gamma = self.gamma(conditioning).unsqueeze(1)
        beta = self.beta(conditioning).unsqueeze(1)
        return gamma * inputs + beta


@dataclass
class DMSTCNOutput:
    logits: Tensor
    attention: Tensor


class DMSTCN(nn.Module):
    def __init__(self, config: DMSTCNConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.branches = nn.ModuleList(
            TemporalBranch(config.hidden_dim, config.kernel_size, schedule, config.dropout)
            for schedule in config.branch_dilations
        )
        self.csag = CrossScaleAttentionGate(config.hidden_dim, len(config.branch_dilations))
        self.film = SubjectFiLM(
            config.num_subjects, config.subject_embedding_dim, config.hidden_dim
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    @property
    def receptive_fields(self) -> tuple[int, ...]:
        return tuple(
            theoretical_receptive_field(self.config.kernel_size, schedule)
            for schedule in self.config.branch_dilations
        )

    def forward(self, inputs: Tensor, subject_ids: Tensor, mask: Tensor | None = None) -> DMSTCNOutput:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, time, features)")
        if subject_ids.shape != (inputs.shape[0],):
            raise ValueError("subject_ids must have shape (batch,)")
        hidden = self.input_projection(inputs)
        branch_outputs = tuple(branch(hidden) for branch in self.branches)
        fused, attention = self.csag(branch_outputs)
        conditioned = self.film(fused, subject_ids)
        if mask is None:
            pooled = conditioned.mean(dim=1)
        else:
            if mask.shape != inputs.shape[:2]:
                raise ValueError("mask must have shape (batch, time)")
            weights = mask.to(conditioned.dtype).unsqueeze(-1)
            lengths = weights.sum(dim=1).clamp_min(1.0)
            pooled = (conditioned * weights).sum(dim=1) / lengths
        return DMSTCNOutput(logits=self.head(pooled), attention=attention)

