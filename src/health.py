"""Leaf segmentation and visible-band crop-health indices.

The project has no multispectral camera, so the health indices are computed
from ordinary RGB images (r, g, b are chromatic coordinates, r = R/(R+G+B)):

    ExG  (Excess Green)                         = 2g - r - b
    VARI (Visible Atmospherically Resistant Ix) = (G - R) / (G + R - B)
    GLI  (Green Leaf Index)                     = (2G - R - B) / (2G + R + B)
    lesion fraction = share of leaf pixels that are not healthy green tissue: hue outside the green band
                      (OpenCV hue 25-100, i.e. ~50-200 degrees) or pale/whitish (S < 30 and V > 170, e.g. mildew)
    health score    = 100 * (1 - lesion fraction)

Indices are averaged over leaf pixels only (background removed).
"""
from __future__ import annotations

import cv2
import numpy as np

GREEN_HUE = (25, 100)  # OpenCV hue range (0-179) treated as healthy green tissue (includes blue-green leaves)
PALE_S, PALE_V = 30, 170  # low-saturation bright pixels = whitish coating (powdery mildew), counted as lesion


def lesion_pixels(hsv: np.ndarray) -> np.ndarray:
    """Boolean lesion map for an HSV image (any shape [..., 3])."""
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= GREEN_HUE[0]) & (h <= GREEN_HUE[1])
    pale = (sat < PALE_S) & (val > PALE_V)
    return ~green | pale


def leaf_mask(rgb: np.ndarray, segmented: np.ndarray | None = None) -> np.ndarray:
    """Boolean leaf mask. Uses the PlantVillage segmented image if given, else HSV thresholding."""
    if segmented is not None:
        m = segmented.astype(np.int32).sum(axis=2) > 30
    else:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        m = (s > 40) & (v > 30)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == biggest


def mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Normalised YOLO box (xc, yc, w, h) around the mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def health_indices(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    px = rgb[mask].astype(np.float64)
    if len(px) == 0:
        return {"exg": float("nan"), "vari": float("nan"), "gli": float("nan"),
                "lesion_fraction": float("nan"), "health_score": float("nan"), "leaf_fraction": 0.0}
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    tot = R + G + B + 1e-6
    r, g, b = R / tot, G / tot, B / tot
    exg = 2 * g - r - b
    vari = np.clip((G - R) / (G + R - B + 1e-6), -1, 1)
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-6)
    lesion = lesion_pixels(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask])
    lf = float(lesion.mean())
    return {
        "exg": float(exg.mean()),
        "vari": float(vari.mean()),
        "gli": float(gli.mean()),
        "lesion_fraction": lf,
        "health_score": 100.0 * (1.0 - lf),
        "leaf_fraction": float(mask.mean()),
    }


def lesion_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB image with lesion pixels highlighted in red (for the app / figures)."""
    lesion = mask & lesion_pixels(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    out = rgb.copy()
    out[lesion] = (0.4 * out[lesion] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    out[~mask] = (out[~mask] * 0.35).astype(np.uint8)
    return out
