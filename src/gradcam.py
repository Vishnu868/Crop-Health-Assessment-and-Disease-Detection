"""Grad-CAM for the classifiers.

    alpha_k^c = (1/Z) sum_ij dy^c / dA^k_ij           (global-average-pooled gradients)
    L^c       = ReLU( sum_k alpha_k^c A^k )            (upsampled to the input size)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def default_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "conv"):          # YoloClassifier: 1x1 head conv (last spatial map)
        return model.conv
    return model.features               # SmallCNN


def grad_cam(model: nn.Module, x: torch.Tensor, target: int | None = None,
             layer: nn.Module | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Return (heatmap HxW in [0,1], predicted/target class, softmax probabilities) for one image (1x3xHxW)."""
    layer = layer or default_target_layer(model)
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        probs = torch.softmax(logits, 1)[0].detach().numpy()
        c = int(logits.argmax(1)) if target is None else target
        model.zero_grad()
        logits[0, c].backward()
        a, g = acts["a"][0], grads["g"][0]
        w = g.mean(dim=(1, 2))
        cam = F.relu((w[:, None, None] * a).sum(0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
    finally:
        h1.remove()
        h2.remove()
    return cam.detach().numpy(), c, probs


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * rgb.astype(np.float32) + alpha * heat, 0, 255).astype(np.uint8)
