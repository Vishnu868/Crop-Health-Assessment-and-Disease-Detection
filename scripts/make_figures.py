#!/usr/bin/env python3
"""Make every figure and summary table used in README.md / REPORT.md from saved results.

    python scripts/make_figures.py
Writes docs/figures/*.png and results/summary.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CLASSES, MODELS, PROCESSED, RAW, RESULTS  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
# validated categorical palette (fixed order) + text inks
PAL = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "axes.titleweight": "bold",
})
SHORT = {"Bacterial_spot": "Bacterial spot", "Black_rot": "Black rot", "Healthy": "Healthy",
         "Late_Blight": "Late blight", "Powder_Mildew": "Powdery mildew"}

CLS_MODELS = [  # (result dir, display name)
    ("yolo26_cls_sfr", "YOLO26n-cls + SFR (proposed)"),
    ("yolo26_cls", "YOLO26n-cls"),
    ("yolo11_cls_sfr", "YOLO11n-cls + SFR"),
    ("yolo11_cls", "YOLO11n-cls"),
    ("yolo26_cls_sfr_scratch", "YOLO26n-cls + SFR, no pretraining"),
    ("small_cnn", "Small CNN (scratch)"),
    ("yolo26_cls_sfr_bgaug", "YOLO26n-cls + SFR, background aug."),
]
SEED_MODELS = [("yolo26_cls_sfr", "YOLO26n-cls + SFR"), ("yolo26_cls", "YOLO26n-cls")]  # seeds 42, 1, 2
DET_MODELS = [("yolo26n_sfr", "YOLO26n + SFR (proposed)"), ("yolo26n", "YOLO26n"),
              ("yolo11n", "YOLO11n"), ("yolov5nu", "YOLOv5n (original arch.)")]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ dataset
def fig_dataset():
    df = pd.read_csv(PROCESSED / "splits.csv")
    fig, axes = plt.subplots(len(CLASSES), 6, figsize=(10, 9))
    rng = np.random.default_rng(3)
    for r, c in enumerate(CLASSES):
        sub = df[(df["class"] == c) & (df.split == "test")]
        srcs = sub.source.unique()
        picks = [sub[sub.source == srcs[k % len(srcs)]].sample(1, random_state=int(rng.integers(1e6))).iloc[0]
                 for k in range(6)]
        for k, row in enumerate(picks):
            img = cv2.cvtColor(cv2.imread(str(RAW / "color" / c / row.file)), cv2.COLOR_BGR2RGB)
            ax = axes[r, k]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(row.source.split("___")[0].replace("_(including_sour)", "").replace(",_bell", " (bell)"),
                         fontsize=8, color=INK2, fontweight="normal")
            if k == 0:
                ax.set_ylabel(SHORT[c], fontsize=10, color=INK)
    fig.suptitle("PlantVillage subset: 5 classes, test-split examples (crop shown above each image)", y=1.0)
    save(fig, "dataset_samples.png")

    # composition: images per class broken down by source crop
    comp = df.groupby(["class", "source"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, c in enumerate(CLASSES):
        left = 0
        for j, (_, row) in enumerate(comp[comp["class"] == c].iterrows()):
            ax.barh(i, row.n - 2, left=left, color=PAL[j % len(PAL)], height=0.6)
            crop = row.source.split("___")[0].split("_(")[0].replace(",_bell", "")
            if row.n > 140:
                ax.text(left + row.n / 2, i, f"{crop}\n{row.n}", ha="center", va="center", fontsize=7, color="white")
            left += row.n
    ax.set_yticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES])
    ax.invert_yaxis()
    ax.set_xlabel("images (train + val + test)")
    ax.set_title("Class composition by source crop (1,600 images per class)")
    ax.grid(axis="y", visible=False)
    save(fig, "dataset_composition.png")


# ------------------------------------------------------------------ health indices
def fig_health():
    df = pd.read_csv(PROCESSED / "splits.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, col, lab in zip(axes, ["health_score", "vari", "exg"],
                            ["Health score (0-100)", "VARI", "Excess Green (ExG)"]):
        data = [df[df["class"] == c][col].dropna() for c in CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5})
        for patch, col_ in zip(bp["boxes"], PAL):
            patch.set_facecolor(col_)
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, 6), [SHORT[c].replace(" ", "\n") for c in CLASSES], fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("RGB crop-health indices over leaf pixels, by class (all 8,000 images)", y=1.03)
    save(fig, "health_indices.png")

    from src.health import leaf_mask, lesion_overlay
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
    for k, c in enumerate(CLASSES):
        row = df[(df["class"] == c) & (df.split == "test")].iloc[3]
        rgb = cv2.cvtColor(cv2.imread(str(RAW / "color" / c / row.file)), cv2.COLOR_BGR2RGB)
        seg_p = RAW / "segmented" / c / row.file
        seg = cv2.cvtColor(cv2.imread(str(seg_p)), cv2.COLOR_BGR2RGB) if seg_p.exists() else None
        m = leaf_mask(rgb, seg)
        for r, im in enumerate([rgb, lesion_overlay(rgb, m)]):
            ax = axes[r, k]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, k].set_title(SHORT[c], fontsize=9)
        axes[1, k].set_xlabel(f"health {row.health_score:.0f} · lesion {100 * row.lesion_fraction:.0f}%", fontsize=8)
    axes[1, 0].set_ylabel("lesion map", fontsize=9)
    axes[0, 0].set_ylabel("input", fontsize=9)
    save(fig, "health_lesion_maps.png")


# ------------------------------------------------------------------ classification
def cls_results():
    out = {}
    for key, name in CLS_MODELS:
        m = load(RESULTS / "classification" / key / "metrics.json")
        if m:
            out[key] = (name, m)
    return out


def plot_cm(ax, cm, labels, title, normalise=True):
    cm = np.asarray(cm, dtype=float)
    disp = cm / cm.sum(1, keepdims=True).clip(min=1) if normalise else cm
    ax.imshow(disp, cmap="Blues", vmin=0, vmax=1)
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if disp[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(disp.shape[0]), labels[:disp.shape[0]], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    ax.grid(False)


def fig_classification(res):
    base = load(RESULTS / "classification" / "baselines_handcrafted" / "metrics.json")
    rows = []
    for key, (name, m) in res.items():
        rows.append((name, m["test"]["accuracy"], m["test"]["macro_f1"]))
    if base:
        rows.append(("Colour+texture features + LogReg", base["logistic_regression"]["accuracy"],
                     base["logistic_regression"]["macro_f1"]))
        rows.append(("Colour+texture features + RandomForest", base["random_forest"]["accuracy"],
                     base["random_forest"]["macro_f1"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [100 * r[1] for r in rows], height=0.36, color=PAL[0], label="accuracy")
    ax.barh(y - 0.19, [100 * r[2] for r in rows], height=0.36, color=PAL[1], label="macro-F1")
    for i, r in enumerate(rows):
        ax.text(100 * r[1] + 0.3, i + 0.19, f"{100 * r[1]:.2f}", va="center", fontsize=8, color=INK)
        ax.text(100 * r[2] + 0.3, i - 0.19, f"{100 * r[2]:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(20, color=INK2, linestyle="--", linewidth=1)
    ax.text(20.5, len(rows) - 0.5, "chance 20%", fontsize=8, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 108)
    ax.set_xlabel("test set (%), 1,200 images")
    ax.set_title("Classifier comparison on the held-out test split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    ax.grid(axis="y", visible=False)
    save(fig, "cls_model_comparison.png")

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (key, (name, m)) in enumerate(res.items()):
        h = pd.DataFrame(m["history"])
        axes[0].plot(h.epoch, h.val_loss, color=PAL[k], linewidth=2, label=name)
        axes[0].plot(h.epoch, h.train_loss, color=PAL[k], linewidth=1, linestyle=":")
        axes[1].plot(h.epoch, 100 * h.val_acc, color=PAL[k], linewidth=2, label=name)
    axes[0].set_title("Loss (solid = validation, dotted = train)", fontsize=10)
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Validation accuracy (%)", fontsize=10)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(50, 100.5)
    axes[0].set_ylim(0, 1.5)
    axes[1].legend(fontsize=7, loc="lower right")
    save(fig, "cls_training_curves.png")

    if "yolo26_cls_sfr" not in res:
        return
    name, m = res["yolo26_cls_sfr"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    plot_cm(axes[0], m["test"]["confusion_matrix"], [SHORT[c] for c in CLASSES], "Original colour test images")
    plot_cm(axes[1], m["test_background_removed"]["confusion_matrix"], [SHORT[c] for c in CLASSES],
            "Same test images, background removed")
    fig.suptitle(f"{name}: confusion matrices (counts; shading = row-normalised recall)", y=1.02)
    save(fig, "cls_confusion_matrix.png")

    # per-class P/R/F1
    pc = m["test"]["per_class"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(CLASSES))
    for k, (met, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1")]):
        vals = [100 * pc[c][met] for c in CLASSES]
        ax.bar(x + (k - 1) * 0.26, vals, width=0.24, color=PAL[k], label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 1) * 0.26, v + 0.15, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    lo = min(100 * pc[c][met] for c in CLASSES for met in ["precision", "recall", "f1-score"])
    ax.set_ylim(max(0, lo - 3), 100.8)
    ax.set_xticks(x, [SHORT[c] for c in CLASSES])
    ax.set_ylabel("%")
    ax.set_title(f"Per-class precision / recall / F1 on the test split: {name}")
    ax.legend(ncols=3, loc="lower right")
    ax.grid(axis="x", visible=False)
    save(fig, "cls_per_class.png")

    # ROC (one-vs-rest)
    p = np.load(RESULTS / "classification" / "yolo26_cls_sfr" / "test_predictions.npz")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    from sklearn.metrics import auc
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(p["y"] == k, p["probs"][:, k])
        ax.plot(fpr, tpr, color=PAL[k], linewidth=2, label=f"{SHORT[c]} (AUC {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], color=INK2, linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("One-vs-rest ROC, test split (log x-axis)", fontsize=10)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "cls_roc.png")


def fig_gradcam():
    ck = MODELS / "yolo26_cls_sfr.pt"
    if not ck.exists():
        return
    from src.data import load_arrays, to_tensor
    from src.gradcam import grad_cam, overlay
    from src.models import load_checkpoint
    model, _ = load_checkpoint(ck)
    X, y, split = load_arrays()
    te = np.where(split == "test")[0]
    fig, axes = plt.subplots(3, 5, figsize=(11, 7))
    rng = np.random.default_rng(0)
    for k, c in enumerate(CLASSES):
        i = int(rng.choice(te[y[te] == k]))
        x = to_tensor(X[i])[None]
        cam, pred, probs = grad_cam(model, x)
        with torch.no_grad():
            model(x)
        sfr_map = model.sfr_modules[-1].last_spatial_map[0, 0].numpy()
        sfr_map = cv2.resize(sfr_map, X[i].shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        sfr_map = (sfr_map - sfr_map.min()) / (np.ptp(sfr_map) + 1e-8)
        for r, im in enumerate([X[i], overlay(X[i], cam), overlay(X[i], sfr_map)]):
            ax = axes[r, k]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, k].set_title(f"true: {SHORT[c]}", fontsize=9)
        axes[1, k].set_xlabel(f"pred: {SHORT[CLASSES[pred]]} ({100 * probs[pred]:.1f}%)", fontsize=8)
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    axes[2, 0].set_ylabel("SFR spatial\nattention (stride 32)", fontsize=9)
    fig.suptitle("Where the YOLO26n-cls + SFR model looks (random correct-class test images)", y=1.0)
    save(fig, "gradcam.png")


# ------------------------------------------------------------------ detection
def det_results():
    out = {}
    for key, name in DET_MODELS:
        m = load(RESULTS / "detection" / key / "metrics.json")
        if m:
            out[key] = (name, m)
    return out


def fig_detection(res):
    if not res:
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    keys = list(res)
    mets = [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95")]
    x = np.arange(len(keys))
    for k, (met, lab) in enumerate(mets):
        vals = [100 * res[key][1]["test"][met] for key in keys]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=PAL[k], label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * 0.36, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    acc = [100 * res[key][1]["test_image_level"]["accuracy"] for key in keys]
    ax.set_xticks(x, [f"{res[k][0]}\nimage-level acc {a:.1f}%" for k, a in zip(keys, acc)], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Detectors on the held-out test split (1,200 images)")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    save(fig, "det_model_comparison.png")

    best = "yolo26n_sfr" if "yolo26n_sfr" in res else keys[0]
    name, m = res[best]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    labs = [SHORT.get(c, "no detection") for c in m["test_image_level"]["confusion_labels"]]
    cm = np.array(m["test_image_level"]["confusion_matrix"])[:len(CLASSES)]
    plot_cm(ax, cm, labs, f"{name}: image-level confusion (top box)")
    save(fig, "det_confusion_matrix.png")

    # sample detections
    from ultralytics import YOLO
    det = YOLO(str(MODELS / f"det_{best}.pt"))
    df = pd.read_csv(PROCESSED / "splits.csv")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4))
    for k, c in enumerate(CLASSES):
        rows = df[(df["class"] == c) & (df.split == "test")].iloc[[5, 11]]
        for r, (_, row) in enumerate(rows.iterrows()):
            bgr = cv2.imread(str(RAW / "color" / c / row.file))
            res_ = det.predict(bgr, imgsz=256, conf=0.25, verbose=False)[0]
            ax = axes[r, k]
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for b, cl, cf in zip(res_.boxes.xyxy.numpy(), res_.boxes.cls.numpy(), res_.boxes.conf.numpy()):
                col = PAL[0] if CLASSES[int(cl)] == c else PAL[7]
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=col, lw=2))
                ax.text(b[0] + 3, b[1] + 12, f"{SHORT[CLASSES[int(cl)]]} {cf:.2f}", fontsize=7, color="white",
                        bbox={"facecolor": col, "edgecolor": "none", "pad": 1.5})
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            if r == 0:
                ax.set_title(f"true: {SHORT[c]}", fontsize=9)
    fig.suptitle(f"{name}: test-split detections (blue = correct class, red = wrong class)", y=1.0)
    save(fig, "det_examples.png")


def seed_results():
    """{key: (name, DataFrame of seed runs)} for the SFR vs no-SFR ablation."""
    out = {}
    for key, name in SEED_MODELS:
        rows = []
        for seed, suffix in [(42, ""), (1, "_seed1"), (2, "_seed2")]:
            m = load(RESULTS / "classification" / f"{key}{suffix}" / "metrics.json")
            if m:
                rows.append({"seed": seed, "acc": 100 * m["test"]["accuracy"], "f1": 100 * m["test"]["macro_f1"],
                             "bg": 100 * m["test_background_removed"]["accuracy"]})
        if rows:
            out[key] = (name, pd.DataFrame(rows))
    return out


def fig_robustness(res, seeds):
    """Test accuracy with the leaf pasted on new backgrounds (shortcut check) + seed scatter for the SFR ablation."""
    rob = load(RESULTS / "classification" / "background_robustness.json")
    if rob:
        variants = [("original", "original photo"), ("black", "black background"),
                    ("flat", "random flat colour"), ("texture", "random colour texture")]
        keys = [k for k, _ in CLS_MODELS if k in rob]
        fig, ax = plt.subplots(figsize=(9, 0.75 * len(keys) + 1.6))
        y = np.arange(len(keys))[::-1]
        hgt = 0.2
        for j, (v, lab) in enumerate(variants):
            vals = [100 * rob[k][v]["accuracy"] for k in keys]
            ax.barh(y + (1.5 - j) * hgt, vals, height=hgt * 0.9, color=PAL[j], label=lab)
            for yi, val in zip(y, vals):
                ax.text(val + 0.4, yi + (1.5 - j) * hgt, f"{val:.1f}", va="center", fontsize=7, color=INK2)
        ax.axvline(20, color=INK2, linestyle="--", linewidth=1)
        ax.text(20.5, len(keys) - 0.45, "chance 20%", fontsize=8, color=INK2)
        ax.set_yticks(y, [dict(CLS_MODELS)[k] for k in keys])
        ax.set_xlim(0, 108)
        ax.set_xlabel("test accuracy (%), same 1,200 leaves, only the background changed")
        ax.set_title("Does the model look at the leaf or at the background?")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4, fontsize=8)
        ax.grid(axis="y", visible=False)
        save(fig, "cls_background_robustness.png")
        ex = np.load(RESULTS / "classification" / "background_examples.npz")
        fig, axes = plt.subplots(4, 6, figsize=(9, 6.2))
        for r, (v, lab) in enumerate(variants):
            for c in range(6):
                axes[r, c].imshow(ex[v][c])
                axes[r, c].axis("off")
            axes[r, 0].set_title(lab, fontsize=9, loc="left")
        fig.suptitle("Background variants used for the robustness test (leaf pixels unchanged)")
        save(fig, "background_variants.png")

    if len(seeds) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, (col, title) in zip(axes, [("acc", "Test accuracy (%)"), ("bg", "Background-removed accuracy (%)")]):
        for k, (key, (name, d)) in enumerate(seeds.items()):
            a.scatter(np.full(len(d), k), d[col], s=60, color=PAL[k], edgecolor="white", linewidth=2, zorder=3)
            a.errorbar(k + 0.18, d[col].mean(), yerr=d[col].std(ddof=1) if len(d) > 1 else 0, fmt="_",
                       color=INK, markersize=14, capsize=4, zorder=2)
            a.text(k + 0.28, d[col].mean(), f"{d[col].mean():.2f}", va="center", fontsize=8, color=INK)
        a.set_xticks(range(len(seeds)), [v[0] for v in seeds.values()], fontsize=8)
        a.set_xlim(-0.5, len(seeds) - 0.2)
        a.set_title(title, fontsize=10)
        a.grid(axis="x", visible=False)
    fig.suptitle("SFR ablation over 3 seeds (dots = runs, bar = mean ± 1 s.d.)", y=1.03)
    save(fig, "cls_sfr_seeds.png")


# ------------------------------------------------------------------ summary table
def write_summary(cres, dres, seeds=None):
    base = load(RESULTS / "classification" / "baselines_handcrafted" / "metrics.json")
    L = ["# Results summary (auto-generated by scripts/make_figures.py)\n",
         "All numbers: held-out **test split, 1,200 real PlantVillage images** (240 per class). Chance = 20%.\n",
         "## Image classification\n",
         "| Model | Pretrained | Params | Accuracy | Macro-F1 | Macro AUC | Acc. (background removed) | Train time |",
         "|---|---|---|---|---|---|---|---|"]
    if base:
        L.append(f"| Majority class | - | - | {100 * base['majority_class']['accuracy']:.2f}% | "
                 f"{100 * base['majority_class']['macro_f1']:.2f}% | - | - | - |")
        for k, n in [("logistic_regression", "Colour+texture + LogReg"), ("random_forest", "Colour+texture + RF")]:
            b = base[k]
            L.append(f"| {n} | - | - | {100 * b['accuracy']:.2f}% | {100 * b['macro_f1']:.2f}% | "
                     f"{b['macro_auc_ovr']:.4f} | - | {b['fit_seconds'] / 60:.1f} min |")
    for key, (name, m) in cres.items():
        t, bg = m["test"], m.get("test_background_removed", {})
        L.append(f"| {name} | {'ImageNet' if m['pretrained'] else 'no'} | {m['params'] / 1e6:.2f} M | "
                 f"{100 * t['accuracy']:.2f}% | {100 * t['macro_f1']:.2f}% | {t['macro_auc_ovr']:.4f} | "
                 f"{100 * bg.get('accuracy', float('nan')):.2f}% | {m['train_minutes']:.1f} min |")
    rob = load(RESULTS / "classification" / "background_robustness.json")
    if rob:
        L += ["\n## Background robustness (same 1,200 test leaves, background replaced)\n",
              "| Model | Original | Black | Random flat colour | Random colour texture |", "|---|---|---|---|---|"]
        for key, name in CLS_MODELS:
            if key in rob:
                r = rob[key]
                L.append(f"| {name} | " + " | ".join(f"{100 * r[v]['accuracy']:.2f}%"
                                                     for v in ["original", "black", "flat", "texture"]) + " |")
    if seeds:
        L += ["\n## SFR ablation over three seeds (42, 1, 2)\n",
              "| Model | Runs | Accuracy (mean ± s.d.) | Macro-F1 | Acc. background removed |", "|---|---|---|---|---|"]
        for key, (name, d) in seeds.items():
            sd = lambda c: d[c].std(ddof=1) if len(d) > 1 else 0.0  # noqa: E731
            L.append(f"| {name} | {len(d)} | {d.acc.mean():.2f} ± {sd('acc'):.2f}% | {d.f1.mean():.2f} ± "
                     f"{sd('f1'):.2f}% | {d.bg.mean():.2f} ± {sd('bg'):.2f}% |")
        L.append("\nIndividual runs: " + "; ".join(
            f"{name} seed {int(r.seed)}: {r.acc:.2f}% / bg-removed {r.bg:.2f}%"
            for _, (name, d) in seeds.items() for r in d.itertuples()) + "\n")
    if dres:
        L += ["\n## Detection (YOLO, auto-generated leaf boxes)\n",
              "| Model | Params | P | R | mAP@0.5 | mAP@0.5:0.95 | Image-level acc. | No detection | Train time |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key, (name, m) in dres.items():
            t, il = m["test"], m["test_image_level"]
            L.append(f"| {name} | {m['params'] / 1e6:.2f} M | {t['precision']:.3f} | {t['recall']:.3f} | "
                     f"{t['mAP50']:.3f} | {t['mAP50_95']:.3f} | {100 * il['accuracy']:.2f}% | {il['no_detection']} | "
                     f"{m['train_minutes']:.0f} min |")
        L += ["\nPer-class detection AP (proposed model):\n", "| Class | P | R | mAP@0.5 | mAP@0.5:0.95 |",
              "|---|---|---|---|---|"]
        best = dres.get("yolo26n_sfr", next(iter(dres.values())))[1]
        for c, v in best["test"]["per_class"].items():
            L.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |")
    if "yolo26_cls_sfr" in cres:
        L += ["\nPer-class classification report (YOLO26n-cls + SFR):\n```",
              (RESULTS / "classification" / "yolo26_cls_sfr" / "classification_report.txt").read_text().rstrip(),
              "```"]
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")
    print("wrote results/summary.md")


def main():
    fig_dataset()
    fig_health()
    cres = cls_results()
    fig_classification(cres)
    seeds = seed_results()
    fig_robustness(cres, seeds)
    fig_gradcam()
    dres = det_results()
    fig_detection(dres)
    write_summary(cres, dres, seeds)


if __name__ == "__main__":
    main()
