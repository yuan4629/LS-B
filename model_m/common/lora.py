"""Naive LoRA module shared by the model_m backbones.

Backbone-agnostic: a low-rank delta added to any nn.Linear's output via a
forward hook. Kept in its own module to avoid pulling in open_clip.
"""
import math

import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA delta: holds only the low-rank A/B; a forward hook adds the delta to the target
    nn.Linear output. delta = (alpha / r) * B(A(x)); A is Kaiming-uniform, B is zero, so
    training starts at delta = 0 (identical to the original weights)."""

    def __init__(self, in_features, out_features, rank=16, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.scaling * self.B(self.A(x))
