"""Dataset loading, normalisation and augmentation (pure torch, CPU friendly)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PROCESSED

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_arrays(path=PROCESSED / "pv128.npz", segmented: bool = False):
    d = np.load(path, allow_pickle=False)
    X = d["Xseg"] if segmented else d["X"]
    return X, d["y"], d["split"]


def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> normalised CHW float tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


def augment(t: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Random flips, 90-degree rotation, small affine (rotate/scale/shift) and colour jitter."""
    if rng.random() < 0.5:
        t = t.flip(2)
    if rng.random() < 0.5:
        t = t.flip(1)
    t = torch.rot90(t, int(rng.integers(4)), dims=(1, 2))
    if rng.random() < 0.7:
        ang = math.radians(rng.uniform(-20, 20))
        sc = rng.uniform(0.85, 1.15)
        tx, ty = rng.uniform(-0.1, 0.1, size=2)
        theta = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, tx],
                              [math.sin(ang) / sc, math.cos(ang) / sc, ty]], dtype=torch.float32)
        grid = torch.nn.functional.affine_grid(theta[None], [1, *t.shape], align_corners=False)
        t = torch.nn.functional.grid_sample(t[None], grid, padding_mode="reflection", align_corners=False)[0]
    # colour jitter in normalised space: brightness / contrast / per-channel gain
    t = t * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2)
    t = t + torch.from_numpy(rng.uniform(-0.1, 0.1, size=(3, 1, 1)).astype(np.float32))
    return t


class LeafDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = False, seed: int = 0,
                 X_alt: np.ndarray | None = None, p_alt: float = 0.0):
        """``X_alt``/``p_alt``: during training, use the alternative (background-removed) image with prob p_alt."""
        self.X, self.y, self.train = X, y, train
        self.X_alt, self.p_alt = X_alt, p_alt
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if not self.train:
            return to_tensor(self.X[i]), int(self.y[i])
        rng = np.random.default_rng((self.seed, self.epoch, i))
        use_alt = self.X_alt is not None and rng.random() < self.p_alt
        t = augment(to_tensor(self.X_alt[i] if use_alt else self.X[i]), rng)
        return t, int(self.y[i])
