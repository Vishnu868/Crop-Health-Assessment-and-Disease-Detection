#!/usr/bin/env python3
"""Diagrams for README / REPORT / guide: system architecture, model + SFR, data pipeline.

    python scripts/make_guide_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, INK2 = "#191919", "#555555"
FILL = {"in": "#dcecfb", "proc": "#e3f5ec", "model": "#fde7d9", "out": "#f3e8fb", "iot": "#fff3d0"}
EDGE = {"in": "#257de0", "proc": "#00a07f", "model": "#f47230", "out": "#6a4ce2", "iot": "#c99000"}


def box(ax, x, y, w, h, text, kind, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=INK2, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=7, color=INK2)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def system():
    fig, ax = canvas(12, 5.2)
    box(ax, 0.2, 3.6, 2.0, 1.0, "Flutter app\n(camera / gallery)", "in")
    box(ax, 0.2, 2.2, 2.0, 1.0, "ESP32-CAM\n(field capture,\nbase64 JPEG)", "in")
    box(ax, 0.2, 0.6, 2.0, 1.0, "Web interface\n(upload)", "in")
    box(ax, 3.0, 2.0, 2.2, 1.6, "Flask REST API\nPOST /predict", "proc", bold=True)
    box(ax, 6.0, 3.3, 2.6, 1.3, "YOLO26n-cls + SFR\nclassifier (128 px)\n→ disease + confidence", "model")
    box(ax, 6.0, 1.8, 2.6, 1.2, "YOLO26n + SFR detector\n(optional, 256 px)\n→ leaf box + class", "model")
    box(ax, 6.0, 0.35, 2.6, 1.15, "RGB health indices\nVARI · ExG · lesion %\n→ health score", "proc")
    box(ax, 9.3, 2.0, 2.5, 1.6, "advisory.json\nfertilizer · pesticide\n· precautions", "out", bold=True)
    box(ax, 9.3, 0.3, 2.5, 1.3, "JSON response\nto app / ESP32 /\nweb page", "out")
    box(ax, 3.0, 4.0, 2.2, 0.95, "NodeMCU + DHT11 +\nsoil moisture + relay", "iot")
    for y in (4.1, 2.7, 1.1):
        arrow(ax, 2.2, y, 3.0, 2.8)
    arrow(ax, 5.2, 2.8, 6.0, 3.95)
    arrow(ax, 5.2, 2.8, 6.0, 2.4)
    arrow(ax, 5.2, 2.8, 6.0, 0.95)
    arrow(ax, 8.6, 3.95, 9.3, 2.8)
    ax.text(9.0, 3.6, "class\nlabel", fontsize=7, color=INK2)
    arrow(ax, 10.55, 2.0, 10.55, 1.6)
    arrow(ax, 8.6, 0.95, 9.3, 0.95)
    arrow(ax, 4.1, 4.0, 4.1, 3.6)
    ax.text(4.2, 3.7, "POST /api/sensors", fontsize=7, color=INK2)
    ax.text(3.0, 5.05, "sensors → pump ON/OFF (hysteresis) + disease-risk from humidity/temperature", fontsize=7.5, color=INK2)
    ax.set_title("AgroDoc system architecture", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "system_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def model():
    fig, ax = canvas(13.2, 4.2)
    stages = [("input\n3×128×128", "in"), ("Conv s2\n16 ch", "proc"), ("Conv s2 +\nC3k2", "proc"),
              ("Conv s2 +\nC3k2", "proc"), ("Conv s2 +\nC3k2\n(128 ch, s16)", "proc"), ("SFR #1", "model"),
              ("Conv s2 +\nC3k2 + C2PSA\n(256 ch, s32)", "proc"), ("SFR #2", "model"),
              ("1×1 conv 1280\n+ GAP + FC", "out"), ("softmax\n5 classes", "out")]
    x = 0.15
    w = 1.12
    for i, (t, k) in enumerate(stages):
        box(ax, x, 2.2, w, 1.3, t, k, fs=7.6, bold=k == "model")
        if i:
            arrow(ax, x - 0.13, 2.85, x, 2.85)
        x += w + 0.13
    ax.text(0.15, 1.75, "ImageNet-pretrained YOLO26n-cls backbone (green) · SFR blocks inserted after the last stride-16 "
            "and stride-32 stages (orange) · new 5-class head (purple)", fontsize=8, color=INK2)
    # SFR detail
    box(ax, 0.6, 0.15, 1.4, 1.0, "X\nC×H×W", "in", fs=8)
    box(ax, 2.4, 0.15, 2.4, 1.0, "channel attention\nMc = σ(MLP(avg X) + MLP(max X))", "model", fs=7.5)
    box(ax, 5.2, 0.15, 3.3, 1.0, "spatial attention\nMs = σ(conv7×7[mean_c; max_c](Mc⊙X))", "model", fs=7.5)
    box(ax, 8.9, 0.15, 3.6, 1.0, "Y = X + γ · (Ms ⊙ Mc ⊙ X)\nγ learned, starts at 0 → identity at init", "out", fs=8)
    arrow(ax, 2.0, 0.65, 2.4, 0.65)
    arrow(ax, 4.8, 0.65, 5.2, 0.65)
    arrow(ax, 8.5, 0.65, 8.9, 0.65)
    ax.set_title("Classifier: YOLO26n-cls + Selective Feature Refinement (SFR)", fontsize=12, fontweight="bold",
                 color=INK)
    fig.savefig(FIG / "model_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def pipeline():
    fig, ax = canvas(12.5, 2.4)
    steps = [("PlantVillage\n(GitHub mirror)\n16 source folders", "in"),
             ("download_plantvillage.py\n1,600 / class, colour +\nsegmented, resumable", "proc"),
             ("preprocess.py\nleaf mask → YOLO box\nhealth indices · 128 px", "proc"),
             ("stratified split\n70 / 15 / 15\n(class × crop)", "proc"),
             ("train_classifier.py\ntrain_detector.py\nbaseline_features.py", "model"),
             ("test metrics, CM, ROC,\nmAP, Grad-CAM,\nbackground-removed test", "out")]
    x, w = 0.15, 1.9
    for i, (t, k) in enumerate(steps):
        box(ax, x, 0.4, w, 1.5, t, k, fs=7.6)
        if i:
            arrow(ax, x - 0.15, 1.15, x, 1.15)
        x += w + 0.15
    ax.set_title("Data and experiment pipeline", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    system()
    model()
    pipeline()
    print("diagrams written to", FIG)
