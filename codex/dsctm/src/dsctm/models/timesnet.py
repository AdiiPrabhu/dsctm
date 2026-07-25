"""TimesNet classification baseline adapted from the official THUML implementation.

Upstream: https://github.com/thuml/Time-Series-Library/blob/
4e938a1767106324dd753b2a44832bf870a0252e/models/TimesNet.py
and ``layers/{Embed,Conv_Blocks}.py`` at the same commit. The temporal 1D→2D
period discovery, inception blocks, adaptive period aggregation, residual path,
embedding, mask application, flattening, and classification projection follow that
code. The constructor is expressed directly in this project's baseline interface.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_for_period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency = xf.abs().mean(0).mean(-1)
    frequency[0] = 0
    top = torch.topk(frequency, min(k, frequency.numel() - 1)).indices
    periods = x.shape[1] // top
    return periods, xf.abs().mean(-1)[:, top]


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for layer in self.kernels:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return torch.stack([layer(x) for layer in self.kernels], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k=2, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels), nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels),
        )

    def forward(self, x):
        B, T, N = x.shape
        periods, weights = fft_for_period(x, self.top_k)
        outputs = []
        for period_tensor in periods:
            period = int(period_tensor.item())
            length = math.ceil(self.seq_len / period) * period
            if length > self.seq_len:
                padding = x.new_zeros(B, length - self.seq_len, N)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2)
            out = self.conv(out.contiguous())
            out = out.permute(0, 2, 3, 1).reshape(B, length, N)
            outputs.append(out[:, :self.seq_len])
        stacked = torch.stack(outputs, dim=-1)
        weights = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        return (stacked * weights).sum(-1) + x


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, 3, padding=1,
                              padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * divisor)
        pe[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class OfficialTimesNetBaseline(nn.Module):
    """Official TimesNet classification pathway with this project's call signature."""
    def __init__(self, input_dim, n_classes, seq_len, d_model=32, d_ff=32,
                 layers=2, top_k=2, num_kernels=6, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.value = TokenEmbedding(input_dim, d_model)
        self.position = PositionalEmbedding(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model * seq_len, n_classes)

    def forward(self, X, subject_idx=None, mask=None):
        if X.shape[1] != self.seq_len:
            raise ValueError(f"TimesNet configured for T={self.seq_len}, got {X.shape[1]}")
        h = self.dropout(self.value(X) + self.position(X))
        for block in self.blocks:
            h = self.norm(block(h))
        h = self.dropout(F.gelu(h))
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
        return self.projection(h.reshape(h.shape[0], -1))
