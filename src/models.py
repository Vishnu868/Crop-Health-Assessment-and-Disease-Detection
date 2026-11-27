"""Classification models.

* ``SmallCNN``         - 4-block CNN trained from scratch (deep-learning baseline)
* ``YoloClassifier``   - ImageNet-pretrained Ultralytics YOLO classification backbone (YOLO26n-cls by
                         default; YOLO11n-cls and YOLOv8n-cls also supported) + optional SFR + new head.
                         ``yolo26_cls_sfr`` is the proposed attention model, ``yolo26_cls`` its ablation.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import torch
from torch import nn

from .config import NUM_CLASSES, PRETRAINED_URLS, WEIGHTS_DIR
from .sfr import SFR


def get_pretrained(name: str) -> Path:
    """Return the local path of a pretrained checkpoint, downloading it from GitHub releases if needed."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    p = WEIGHTS_DIR / name
    if not p.exists() or p.stat().st_size == 0:
        urllib.request.urlretrieve(PRETRAINED_URLS[name], p)
    return p


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 24):
        super().__init__()
        layers, c = [], 3
        for i in range(4):
            o = width * 2 ** i
            layers += [nn.Conv2d(c, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = o
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class YoloClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, sfr: bool = True, pretrained: bool = True,
                 arch: str = "yolo26n-cls", sfr_stages: tuple[int, ...] | None = None):
        super().__init__()
        from ultralytics.nn.tasks import ClassificationModel

        if pretrained:
            from ultralytics import YOLO
            base = YOLO(str(get_pretrained(f"{arch}.pt"))).model
        else:
            base = ClassificationModel(f"{arch.replace('n-cls', 'n-cls')}.yaml", nc=1000, verbose=False)
        if sfr_stages is None:  # refine the last stride-16 block and the last stride-32 block
            sfr_stages = (6, 8) if arch.startswith("yolov8") else (6, 9)
        layers = list(base.model.children())
        head = layers[-1]
        blocks = []
        self.sfr_modules = nn.ModuleList()
        for i, layer in enumerate(layers[:-1]):
            blocks.append(layer)
            if sfr and i in sfr_stages:
                ch = _out_channels(layer)
                m = SFR(ch)
                blocks.append(m)
                self.sfr_modules.append(m)
        self.features = nn.Sequential(*blocks)
        self.conv = head.conv            # pretrained 1x1 conv 256 -> 1280
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(head.linear.in_features, num_classes)
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        x = self.conv(self.features(x))
        return self.fc(self.drop(self.pool(x).flatten(1)))


def _out_channels(layer: nn.Module) -> int:
    convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]
    # C2f ends with cv2 (the output conv); Conv has a single conv
    last = getattr(layer, "cv2", None)
    if last is not None and hasattr(last, "conv"):
        return last.conv.out_channels
    return convs[-1].out_channels


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    m = re.fullmatch(r"(yolo26|yolo11|yolov8)_cls(_sfr)?", name)
    if m:
        return YoloClassifier(sfr=bool(m.group(2)), pretrained=pretrained, arch=f"{m.group(1)}n-cls")
    raise ValueError(f"unknown model {name}")


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_name"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
