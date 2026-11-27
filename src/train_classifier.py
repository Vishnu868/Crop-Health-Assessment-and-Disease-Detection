#!/usr/bin/env python3
"""Train and evaluate a leaf-disease classifier on the 5-class PlantVillage subset.

    python -m src.train_classifier --model yolo26_cls_sfr --epochs 12
    python -m src.train_classifier --model small_cnn --epochs 15
    python -m src.train_classifier --data synthetic --model yolo26_cls_sfr --epochs 2   # sanity check

Loss: label-smoothed cross-entropy; optimiser AdamW; cosine LR with warm-up;
model selection on validation macro-F1; final numbers on the held-out test split.
Also evaluates the best model on background-removed test images (shortcut check).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, MODELS, RESULTS, SEED  # noqa: E402
from src.data import LeafDataset, load_arrays  # noqa: E402
from src.models import build_model  # noqa: E402


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


def synthetic_arrays(n_per_class: int = 120, size: int = 128, seed: int = 0):
    """Synthetic 'leaves' (SANITY CHECK ONLY): a green ellipse with class-specific coloured spots."""
    rng = np.random.default_rng(seed)
    spot_colours = [(70, 50, 20), (20, 15, 15), None, (90, 80, 60), (235, 235, 225)]
    X, y = [], []
    yy, xx = np.mgrid[:size, :size]
    for c, col in enumerate(spot_colours):
        for _ in range(n_per_class):
            img = np.full((size, size, 3), rng.integers(150, 200), np.uint8)
            cx, cy = rng.uniform(0.4, 0.6, 2) * size
            leaf = ((xx - cx) / (0.35 * size)) ** 2 + ((yy - cy) / (0.25 * size)) ** 2 < 1
            img[leaf] = (rng.integers(40, 70), rng.integers(120, 170), rng.integers(30, 60))
            if col is not None:
                for _ in range(rng.integers(4, 10)):
                    sx, sy, r = rng.uniform(0.25, 0.75) * size, rng.uniform(0.3, 0.7) * size, rng.uniform(2, 6)
                    img[((xx - sx) ** 2 + (yy - sy) ** 2 < r * r) & leaf] = col
            X.append(img)
            y.append(c)
    X, y = np.stack(X), np.array(y)
    split = rng.choice(["train", "val", "test"], size=len(y), p=[0.7, 0.15, 0.15])
    return X, y, split


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), 1).numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(),
           "per_class": classification_report(y, pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                              output_dict=True, zero_division=0)}
    try:
        out["macro_auc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auc_ovr"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26_cls_sfr", choices=["small_cnn", "yolo26_cls", "yolo26_cls_sfr", "yolo11_cls", "yolo11_cls_sfr", "yolov8_cls"])
    ap.add_argument("--data", default="real", choices=["real", "synthetic"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--train-frac", type=float, default=1.0, help="use a fraction of the training split")
    ap.add_argument("--bg-aug", type=float, default=0.0,
                    help="probability of training on the background-removed version of an image")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    if args.data == "real":
        X, y, split = load_arrays()
        Xseg, _, _ = load_arrays(segmented=True)
    else:
        X, y, split = synthetic_arrays(seed=args.seed)
        Xseg = None
    tr = np.where(split == "train")[0]
    if args.train_frac < 1:
        tr = np.random.default_rng(args.seed).choice(tr, int(len(tr) * args.train_frac), replace=False)
    va, te = np.where(split == "val")[0], np.where(split == "test")[0]
    name = f"{args.model}{'_scratch' if args.no_pretrained else ''}{'_bgaug' if args.bg_aug > 0 else ''}{args.tag}"
    if args.data == "synthetic":
        name = f"synthetic_{name}"
    out_dir = RESULTS / "classification" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name}: train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)

    train_ds = LeafDataset(X[tr], y[tr], train=True, seed=args.seed,
                           X_alt=Xseg[tr] if (Xseg is not None and args.bg_aug > 0) else None, p_alt=args.bg_aug)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(LeafDataset(X[va], y[va]), batch_size=128)
    test_dl = DataLoader(LeafDataset(X[te], y[te]), batch_size=128)

    model = build_model(args.model, pretrained=not args.no_pretrained)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best_f1, best_state = [], -1.0, None
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        train_ds.epoch = ep
        tot, correct, loss_sum = 0, 0, 0.0
        t0 = time.time()
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(yb)
        vp, vy = predict(model, val_dl)
        vloss = float(nn.functional.cross_entropy(torch.log(torch.from_numpy(vp) + 1e-9), torch.from_numpy(vy)))
        vm = metrics(vp, vy)
        h = {"epoch": ep + 1, "train_loss": loss_sum / tot, "train_acc": correct / tot, "val_loss": vloss,
             "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "seconds": time.time() - t0}
        history.append(h)
        print(json.dumps(h), flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tp, ty = predict(model, test_dl)
    res = {"model": args.model, "data": args.data, "pretrained": not args.no_pretrained,
           "params": n_params, "bg_aug": args.bg_aug, "epochs": args.epochs, "train_size": len(tr), "val_size": len(va),
           "test_size": len(te), "train_minutes": (time.time() - t_start) / 60, "best_val_macro_f1": best_f1,
           "test": metrics(tp, ty), "history": history}
    if Xseg is not None:
        sp, sy = predict(model, DataLoader(LeafDataset(Xseg[te], y[te]), batch_size=128))
        res["test_background_removed"] = metrics(sp, sy)
    if args.model.endswith("_sfr"):
        res["sfr_gamma"] = [float(m.gamma.detach()) for m in model.sfr_modules]
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    (out_dir / "classification_report.txt").write_text(
        classification_report(ty, tp.argmax(1), labels=range(len(CLASSES)), target_names=CLASSES, digits=4))
    np.savez_compressed(out_dir / "test_predictions.npz", probs=tp, y=ty, idx=te)
    if args.data == "real":
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": args.model, "state_dict": model.state_dict(), "classes": CLASSES,
                    "img_size": X.shape[1]}, MODELS / f"{name}.pt")
    print(f"TEST acc={res['test']['accuracy']:.4f} macroF1={res['test']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
