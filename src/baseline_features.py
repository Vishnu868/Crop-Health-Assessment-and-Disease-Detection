#!/usr/bin/env python3
"""Classical baseline: colour + texture features -> logistic regression / random forest.

Features per image (128x128):
  * HSV colour histogram (16 hue x 4 sat x 4 val = 256 bins, L1-normalised)
  * HOG on the grey image (16 px cells, 9 unsigned orientations -> 8x8x9 = 576 values)
  * Local Binary Pattern histogram (uniform, P=8, R=1 -> 10 bins)
Also reports the chance level and the majority-class baseline.

    python -m src.baseline_features
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, RESULTS, SEED  # noqa: E402
from src.data import load_arrays  # noqa: E402
from src.train_classifier import metrics  # noqa: E402



def hog(gray: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients: unsigned orientation, magnitude-weighted, per-cell L2-normalised."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    b = np.minimum((ang / (180.0 / bins)).astype(np.int32), bins - 1)
    H, W = g.shape
    ny, nx = H // cell, W // cell
    out = np.zeros((ny, nx, bins), np.float32)
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            out[i, j] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=bins)
    out /= np.linalg.norm(out, axis=2, keepdims=True) + 1e-6
    return out.ravel()


def lbp_hist(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    nb = [g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2]]
    bits = np.stack([(n >= c) for n in nb]).astype(np.uint8)
    transitions = np.abs(np.diff(np.concatenate([bits, bits[:1]]), axis=0)).sum(0)
    ones = bits.sum(0)
    code = np.where(transitions <= 2, ones, 9)  # uniform patterns 0..8, non-uniform -> 9
    h = np.bincount(code.ravel(), minlength=10).astype(np.float32)
    return h / h.sum()


def features(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    h /= h.sum() + 1e-9
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.concatenate([h, hog(gray), lbp_hist(gray)])


def main() -> None:
    X, y, split = load_arrays()
    t0 = time.time()
    F = np.stack([features(x) for x in X]).astype(np.float32)
    print(f"features {F.shape} in {time.time() - t0:.0f}s")
    tr, te = split == "train", split == "test"
    out = {}
    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=400, n_jobs=2, random_state=SEED),
    }
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(F[tr], y[tr])
        probs = clf.predict_proba(F[te])
        m = metrics(probs, y[te])
        m["fit_seconds"] = time.time() - t0
        out[name] = m
        print(f"{name}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
    out["chance_level"] = 1 / len(CLASSES)
    out["feature_dim"] = int(F.shape[1])
    d = RESULTS / "classification" / "baselines_handcrafted"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
