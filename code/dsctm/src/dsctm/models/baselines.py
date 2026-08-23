"""The seven baselines (manuscript §V.B / Table 2). All share the mean-pool + MLP
head and expose a uniform `forward(X, subject_idx=None)` interface (subject_idx is
ignored — only D-MSTCN personalizes).

DataParallel-LSTM and FedAvg-LSTM share the LSTM *architecture*; they differ only
in the distributed training protocol (applied in the training loop, not here).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Branch, Head
from .timesnet import OfficialTimesNetBaseline

BASELINES = [
    "lstm", "temporal-cnn", "transformer", "timesnet",
    "itransformer", "dataparallel-lstm", "fedavg-lstm",
]


class LSTMBaseline(nn.Module):
    """Bidirectional 2-layer LSTM, hidden 128."""

    def __init__(self, input_dim, n_classes, hidden=128, layers=2, bidirectional=True,
                 head_hidden=128, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = Head(hidden * (2 if bidirectional else 1), n_classes, head_hidden)

    def forward(self, X, subject_idx=None, mask=None):
        if mask is None:
            h, _ = self.lstm(X)
        else:
            lengths = mask.sum(1).to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                X, lengths, batch_first=True, enforce_sorted=False
            )
            packed_h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(
                packed_h, batch_first=True, total_length=X.size(1)
            )
        return self.head(h, mask)


class TCNBaseline(nn.Module):
    """Temporal-CNN: dilated causal TCN, 4 residual blocks."""

    def __init__(self, input_dim, n_classes, D=128, K=3, dilations=(1, 2, 4, 8),
                 head_hidden=128, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, D)
        self.tcn = Branch(D, K, dilations, dropout=dropout)
        self.head = Head(D, n_classes, head_hidden)

    def forward(self, X, subject_idx=None, mask=None):
        h = self.proj(X).transpose(1, 2)
        return self.head(self.tcn(h).transpose(1, 2), mask)


class TransformerBaseline(nn.Module):
    """Encoder Transformer: d_model=128, 4 heads, 2 layers, sinusoidal positions."""

    def __init__(self, input_dim, n_classes, d_model=128, nhead=4, layers=2,
                 head_hidden=128, max_len=5000, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.register_buffer("pos", self._sinusoid(max_len, d_model), persistent=False)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = Head(d_model, n_classes, head_hidden)

    @staticmethod
    def _sinusoid(L, D):
        pe = torch.zeros(L, D)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, X, subject_idx=None, mask=None):
        h = self.proj(X)
        h = h + self.pos[: h.size(1)].unsqueeze(0)
        return self.head(self.enc(h, src_key_padding_mask=(~mask if mask is not None else None)), mask)


class ITransformerBaseline(nn.Module):
    """iTransformer: embed each variate's whole series as one token, attend across
    variates (inverted attention). Simplified reimplementation (fixed T via LazyLinear)."""

    def __init__(self, input_dim, n_classes, d_model=128, nhead=4, layers=2,
                 head_hidden=128, dropout=0.1):
        super().__init__()
        self.embed = nn.LazyLinear(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = Head(d_model, n_classes, head_hidden)

    def forward(self, X, subject_idx=None, mask=None):
        if mask is not None:
            X = X.masked_fill(~mask.unsqueeze(-1), 0.0)
        tok = self.embed(X.transpose(1, 2))   # (B, F, d_model) — one token per variate
        return self.head(self.enc(tok))       # head mean-pools over variate tokens


class TimesNetBaseline(nn.Module):
    """SIMPLIFIED TimesNet (inception-style multi-kernel temporal conv).
    NOTE: this is a placeholder, not the full FFT period-folding TimesNet — replace
    with the faithful implementation before making EXP-2.2 fair-baseline claims."""

    def __init__(self, input_dim, n_classes, d_model=128, head_hidden=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.branches = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.head = Head(d_model, n_classes, head_hidden)

    def forward(self, X, subject_idx=None, mask=None):
        h = self.proj(X).transpose(1, 2)
        h = sum(F.gelu(b(h)) for b in self.branches).transpose(1, 2)
        return self.head(h, mask)


def build_baseline(name: str, input_dim: int, n_classes: int, head_hidden: int = 128,
                   seq_len: int | None = None) -> nn.Module:
    name = name.lower()
    if name in ("lstm", "dataparallel-lstm", "fedavg-lstm"):
        return LSTMBaseline(input_dim, n_classes, head_hidden=head_hidden)
    if name in ("tcn", "temporal-cnn"):
        return TCNBaseline(input_dim, n_classes, head_hidden=head_hidden)
    if name == "transformer":
        return TransformerBaseline(input_dim, n_classes, head_hidden=head_hidden)
    if name == "itransformer":
        return ITransformerBaseline(input_dim, n_classes, head_hidden=head_hidden)
    if name == "timesnet":
        if seq_len is None:
            raise ValueError("faithful TimesNet requires the fixed input sequence length")
        return OfficialTimesNetBaseline(input_dim, n_classes, seq_len=seq_len)
    if name == "timesnet-simplified":
        return TimesNetBaseline(input_dim, n_classes, head_hidden=head_hidden)
    raise KeyError(f"unknown baseline {name!r}; have {BASELINES}")
