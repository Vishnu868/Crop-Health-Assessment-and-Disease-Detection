#!/usr/bin/env python3
"""Inference: leaf image -> disease, confidence, advisory, health indices (+ optional Grad-CAM / boxes).

    python -m src.predict path/to/leaf.jpg
    python -m src.predict leaf.jpg --gradcam cam.png --detector results/models/det_yolo26n_sfr.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, IMG_SIZE, MODELS, ROOT  # noqa: E402
from src.data import to_tensor  # noqa: E402
from src.health import health_indices, leaf_mask  # noqa: E402

DEFAULT_CLASSIFIER = MODELS / "yolo26_cls_sfr.pt"
ADVISORY_PATH = ROOT / "app" / "advisory.json"


@lru_cache(maxsize=1)
def load_advisory() -> dict:
    return json.loads(ADVISORY_PATH.read_text())


@lru_cache(maxsize=4)
def load_classifier(path: str = str(DEFAULT_CLASSIFIER)):
    from src.models import load_checkpoint
    model, ck = load_checkpoint(path)
    return model, ck.get("img_size", IMG_SIZE)


@lru_cache(maxsize=2)
def load_detector(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def read_rgb(src) -> np.ndarray:
    """Accept a path, raw bytes or an RGB array."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def predict(src, classifier: str = str(DEFAULT_CLASSIFIER), detector: str | None = None,
            gradcam_path: str | None = None) -> dict:
    rgb = read_rgb(src)
    model, size = load_classifier(classifier)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = to_tensor(small)[None]
    if gradcam_path:
        from src.gradcam import grad_cam, overlay
        cam, _, probs = grad_cam(model, x)
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay(small, cam), cv2.COLOR_RGB2BGR))
    else:
        with torch.no_grad():
            probs = torch.softmax(model(x), 1)[0].numpy()
    order = np.argsort(probs)[::-1]
    label = CLASSES[int(order[0])]
    adv = load_advisory()[label]
    health = health_indices(rgb, leaf_mask(rgb))
    out = {
        "disease": adv["disease"],
        "class": label,
        "confidence": round(float(probs[order[0]]), 4),
        "top3": [{"class": CLASSES[int(i)], "prob": round(float(probs[i]), 4)} for i in order[:3]],
        "fertilizer": adv["fertilizer"],
        "pesticide": adv["pesticide"],
        "precautions": adv["precautions"],
        "symptoms": adv["symptoms"],
        "health": {k: round(v, 4) for k, v in health.items()},
        "low_confidence": bool(probs[order[0]] < 0.6),
    }
    if detector:
        r = load_detector(detector).predict(rgb[..., ::-1].copy(), imgsz=256, conf=0.25, verbose=False)[0]
        out["detections"] = [{"class": CLASSES[int(c)], "conf": round(float(p), 4),
                              "box_xyxy": [round(float(v), 1) for v in b]}
                             for b, c, p in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER))
    ap.add_argument("--detector", default=None)
    ap.add_argument("--gradcam", default=None, help="write a Grad-CAM overlay PNG here")
    a = ap.parse_args()
    print(json.dumps(predict(a.image, a.classifier, a.detector, a.gradcam), indent=2))


if __name__ == "__main__":
    main()
