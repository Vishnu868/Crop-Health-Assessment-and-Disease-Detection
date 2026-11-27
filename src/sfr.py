"""Selective Feature Refinement (SFR) attention module.

SFR refines a feature map X (C x H x W) in two selective steps and adds the
result back through a learnable gate, so that inserting it into a pretrained
network does not disturb the network at initialisation (gamma = 0):

    channel weights   M_c = sigmoid( MLP(AvgPool(X)) + MLP(MaxPool(X)) )            (C x 1 x 1)
    X'                = M_c * X
    spatial weights   M_s = sigmoid( Conv7x7( [mean_c(X'); max_c(X')] ) )            (1 x H x W)
    X''               = M_s * X'
    output            Y   = X + gamma * X''

The channel step selects *which* feature detectors matter (e.g. "brown lesion"
filters), the spatial step selects *where* on the leaf to look. ``gamma`` is a
learned scalar initialised to 0.
"""
from __future__ import annotations

import torch
from torch import nn


class SFR(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_spatial_map: torch.Tensor | None = None  # kept for visualisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        mc = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        xc = x * mc
        s = torch.cat([xc.mean(1, keepdim=True), xc.amax(1, keepdim=True)], dim=1)
        ms = torch.sigmoid(self.spatial(s))
        self.last_spatial_map = ms.detach()
        return x + self.gamma * (xc * ms)


class SFRWrap(nn.Module):
    """Wrap an existing layer so its output is refined by SFR (keeps ultralytics' i/f/type attributes)."""

    def __init__(self, layer: nn.Module, channels: int):
        super().__init__()
        self.layer = layer
        self.sfr = SFR(channels)
        for attr in ("i", "f", "type", "np"):
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

    def forward(self, x):
        return self.sfr(self.layer(x))
