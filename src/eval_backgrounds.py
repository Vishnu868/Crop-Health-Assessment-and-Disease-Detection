#!/usr/bin/env python3
"""Background-robustness test: paste every test leaf onto backgrounds no model saw in training.

    python -m src.eval_backgrounds

For each saved classifier (results/models/<name>.pt) the 1,200 test leaves are evaluated with
  original   - the PlantVillage photo as is
  black      - PlantVillage's background-removed image (identical to `test_background_removed`)
  flat       - background replaced by one random uniform colour per image
  texture    - background replaced by random low-frequency colour noise (blotchy, soil/foliage-like)
The leaf pixels themselves are never changed. Writes results/classification/background_robustness.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS, RESULTS, SEED  # noqa: E402
from src.data import LeafDataset, load_arrays  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.train_classifier import metrics, predict  # noqa: E402

MODEL_KEYS = ["yolo26_cls_sfr", "yolo26_cls", "yolo11_cls_sfr", "yolo11_cls", "yolo26_cls_sfr_scratch",
              "small_cnn", "yolo26_cls_sfr_bgaug"]


def backgrounds(X: np.ndarray, Xseg: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = (Xseg.max(-1) > 8)[..., None]  # leaf pixels (the segmented image is black outside the leaf)
    n, h, w, _ = X.shape
    flat = np.broadcast_to(rng.integers(0, 256, (n, 1, 1, 3), dtype=np.uint8), X.shape)
    tex = np.empty_like(X)
    for i in range(n):
        coarse = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
        t = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.int16)
        t += rng.integers(-25, 26, (h, w, 3), dtype=np.int16)
        tex[i] = np.clip(t, 0, 255).astype(np.uint8)
    return {"original": X, "black": Xseg,  # exactly the `test_background_removed` images
            "flat": np.where(mask, X, flat).astype(np.uint8), "texture": np.where(mask, X, tex).astype(np.uint8)}


def main() -> None:
    torch.set_num_threads(4)
    X, y, split = load_arrays()
    Xseg, _, _ = load_arrays(segmented=True)
    te = split == "test"
    variants = backgrounds(X[te], Xseg[te])
    out = {}
    for key in MODEL_KEYS:
        path = MODELS / f"{key}.pt"
        if not path.exists():
            continue
        model, _ = load_checkpoint(path)
        out[key] = {}
        for v, Xv in variants.items():
            p, yy = predict(model, DataLoader(LeafDataset(Xv, y[te]), batch_size=128))
            m = metrics(p, yy)
            out[key][v] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                           "confusion_matrix": m["confusion_matrix"]}
        print(key, {v: round(100 * r["accuracy"], 2) for v, r in out[key].items()}, flush=True)
    (RESULTS / "classification" / "background_robustness.json").write_text(json.dumps(out, indent=1))
    # one example strip per variant for the figure
    np.savez_compressed(RESULTS / "classification" / "background_examples.npz",
                        **{v: Xv[::150][:8] for v, Xv in variants.items()})


if __name__ == "__main__":
    main()
