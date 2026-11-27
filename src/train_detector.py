#!/usr/bin/env python3
"""Train / evaluate a YOLO leaf-disease detector (Ultralytics), optionally with SFR attention.

    python -m src.train_detector --arch yolo26n --sfr --epochs 20        # proposed: YOLO26n + SFR
    python -m src.train_detector --arch yolo26n --epochs 20              # ablation without SFR
    python -m src.train_detector --arch yolo11n --epochs 20
    python -m src.train_detector --arch yolov5nu --epochs 20             # original-project architecture

SFR blocks are inserted on the three neck outputs (P3/P4/P5) that feed the
Detect head. Each block starts as an identity (gamma = 0), so the COCO-
pretrained weights are used unchanged at the start of fine-tuning.

Evaluation on the held-out test split reports P, R, mAP50, mAP50-95 per class,
and an image-level accuracy/confusion matrix (class of the most confident box;
"no detection" counts as an error) so detectors and classifiers can be compared.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, DET_IMG_SIZE, MODELS, RESULTS, SEED, YOLO_DIR  # noqa: E402
from src.models import get_pretrained  # noqa: E402
from src.sfr import SFRWrap  # noqa: E402


def add_sfr(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap every layer that feeds the Detect head with an SFR block (in place)."""
    detect = model.model[-1]
    for k, src_idx in enumerate(detect.f):
        ch = detect.cv2[k][0].conv.in_channels
        layer = model.model[src_idx]
        if not isinstance(layer, SFRWrap):
            model.model[src_idx] = SFRWrap(layer, ch)
    return model


def make_trainer(use_sfr: bool):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class SFRTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            return add_sfr(model) if use_sfr else model

    return SFRTrainer


def image_level_eval(weights: Path, split: str = "test", conf: float = 0.25) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    img_dir = YOLO_DIR / "images" / split
    files = sorted(img_dir.iterdir())
    y_true, y_pred, n_none = [], [], 0
    for i in range(0, len(files), 64):
        batch = files[i:i + 64]
        for f, r in zip(batch, model.predict([str(p) for p in batch], imgsz=DET_IMG_SIZE, conf=conf,
                                             verbose=False)):
            lab = int((YOLO_DIR / "labels" / split / (f.stem + ".txt")).read_text().split()[0])
            y_true.append(lab)
            if len(r.boxes) == 0:
                y_pred.append(len(CLASSES))  # "no detection"
                n_none += 1
            else:
                y_pred.append(int(r.boxes.cls[r.boxes.conf.argmax()]))
    labels = list(range(len(CLASSES) + 1))
    names = CLASSES + ["no_detection"]
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=range(len(CLASSES)), average="macro")),
            "no_detection": n_none, "conf_threshold": conf,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "confusion_labels": names,
            "per_class": classification_report(y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                               output_dict=True, zero_division=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="yolo26n", choices=["yolo26n", "yolo11n", "yolov5nu"])
    ap.add_argument("--sfr", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=DET_IMG_SIZE)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the training split to use")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from ultralytics import YOLO

    name = f"{args.arch}{'_sfr' if args.sfr else ''}"
    out_dir = RESULTS / "detection" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = Path("runs") / "detect"
    t0 = time.time()
    model = YOLO(str(get_pretrained(f"{args.arch}.pt")))
    model.train(data=str(YOLO_DIR / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device="cpu", workers=args.workers, seed=SEED, deterministic=False, fraction=args.fraction,
                project=str(runs.resolve()), name=name, exist_ok=True, plots=True, patience=100,
                optimizer="AdamW", lr0=0.002, cos_lr=True, close_mosaic=max(1, args.epochs // 4),
                trainer=make_trainer(args.sfr), amp=False, verbose=False)
    train_min = (time.time() - t0) / 60
    run_dir = runs / name
    best = run_dir / "weights" / "best.pt"

    val = YOLO(str(best)).val(data=str(YOLO_DIR / "data.yaml"), split="test", imgsz=args.imgsz, batch=64,
                              device="cpu", plots=True, project=str(runs.resolve()), name=f"{name}_test",
                              exist_ok=True, verbose=False)
    per_class = {}
    for k, c in enumerate(CLASSES):
        p, r, ap50, ap = val.box.class_result(k)
        per_class[c] = {"precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50_95": float(ap)}
    res = {"arch": args.arch, "sfr": args.sfr, "epochs": args.epochs, "imgsz": args.imgsz,
           "train_fraction": args.fraction, "train_minutes": train_min,
           "params": int(sum(p.numel() for p in YOLO(str(best)).model.parameters())),
           "test": {"precision": float(val.box.mp), "recall": float(val.box.mr), "mAP50": float(val.box.map50),
                    "mAP50_95": float(val.box.map), "per_class": per_class},
           "test_image_level": image_level_eval(best)}
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    for f in ["results.csv", "results.png"]:
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out_dir / f)
    test_dir = runs / f"{name}_test"
    for f in ["confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"]:
        if (test_dir / f).exists():
            shutil.copy(test_dir / f, out_dir / f)
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODELS / f"det_{name}.pt")
    print(json.dumps({k: res["test"][k] for k in ["precision", "recall", "mAP50", "mAP50_95"]}),
          "image-level acc", res["test_image_level"]["accuracy"], f"{train_min:.1f} min")


if __name__ == "__main__":
    main()
