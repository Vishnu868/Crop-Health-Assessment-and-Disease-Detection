"""Fast unit tests (no dataset download needed). Run: pytest -q"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASSES  # noqa: E402
from src.data import LeafDataset, augment, to_tensor  # noqa: E402
from src.health import health_indices, leaf_mask, mask_to_box  # noqa: E402
from src.irrigation import IrrigationController, adc_to_moisture, disease_risk  # noqa: E402
from src.models import SmallCNN, build_model  # noqa: E402
from src.sfr import SFR, SFRWrap  # noqa: E402


def fake_leaf(spots: bool = False, size: int = 128) -> np.ndarray:
    img = np.full((size, size, 3), 180, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 4), 0, 0, 360, (50, 150, 40), -1)
    if spots:
        for c in [(50, 60), (70, 64), (60, 75)]:
            cv2.circle(img, c, 5, (90, 60, 20), -1)
    return img


# ---------------------------------------------------------------- SFR
def test_sfr_is_identity_at_init():
    m = SFR(32)
    x = torch.randn(2, 32, 8, 8)
    assert torch.allclose(m(x), x)


def test_sfr_changes_output_when_gamma_nonzero_and_has_grads():
    m = SFR(16)
    with torch.no_grad():
        m.gamma.fill_(1.0)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and not torch.allclose(y, x)
    y.sum().backward()
    assert m.spatial.weight.grad is not None and m.last_spatial_map.shape == (2, 1, 8, 8)


def test_sfrwrap_keeps_ultralytics_attributes():
    layer = torch.nn.Conv2d(3, 8, 3, padding=1)
    layer.i, layer.f, layer.type = 5, -1, "Conv"
    w = SFRWrap(layer, 8)
    assert (w.i, w.f, w.type) == (5, -1, "Conv")
    assert w(torch.randn(1, 3, 16, 16)).shape == (1, 8, 16, 16)


# ---------------------------------------------------------------- models
def test_small_cnn_shapes():
    assert SmallCNN()(torch.randn(2, 3, 128, 128)).shape == (2, len(CLASSES))


@pytest.mark.parametrize("name", ["yolo26_cls", "yolo26_cls_sfr"])
def test_yolo_classifier_shapes(name):
    m = build_model(name, pretrained=False)
    assert m(torch.randn(2, 3, 128, 128)).shape == (2, len(CLASSES))
    if name.endswith("_sfr"):
        assert len(m.sfr_modules) == 2


def test_detector_sfr_insertion_preserves_output():
    from ultralytics.nn.tasks import DetectionModel

    from src.train_detector import add_sfr
    torch.manual_seed(0)
    m = DetectionModel("yolo26n.yaml", nc=len(CLASSES), verbose=False).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = m(x)[0].clone()
        add_sfr(m)
        after = m(x)[0]
    assert sum(type(l).__name__ == "SFRWrap" for l in m.model) == 3
    assert torch.allclose(before, after, atol=1e-5)  # gamma = 0 -> identical predictions


# ---------------------------------------------------------------- data
def test_to_tensor_and_augment():
    t = to_tensor(fake_leaf())
    assert t.shape == (3, 128, 128) and t.dtype == torch.float32
    a = augment(t, np.random.default_rng(0))
    assert a.shape == t.shape and torch.isfinite(a).all()


def test_dataset_deterministic_augmentation():
    X = np.stack([fake_leaf(), fake_leaf(True)])
    ds = LeafDataset(X, np.array([2, 0]), train=True, seed=1)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)  # same (seed, epoch, index) -> same augmentation
    ds.epoch = 1
    c, _ = ds[0]
    assert not torch.equal(a, c)


def test_balanced_sample():
    sys.path.insert(0, str(ROOT / "scripts"))
    from download_plantvillage import CLASS_MAP, balanced_sample
    tree = {s: [f"{s}_{i}.JPG" for i in range(50 if "Apple" in s else 500)] for v in CLASS_MAP.values() for s in v}
    rows = balanced_sample(tree, 300, seed=0)
    counts = {c: sum(r[0] == c for r in rows) for c in CLASS_MAP}
    assert all(v == 300 for v in counts.values())
    apple_br = sum(r[1] == "Apple___Black_rot" for r in rows)
    assert apple_br == 50  # small source fully used, the rest filled by Grape
    assert len({r[2] for r in rows}) == len(rows)  # no duplicates


# ---------------------------------------------------------------- health indices
def test_leaf_mask_and_box():
    img = fake_leaf()
    m = leaf_mask(img)
    assert 0.15 < m.mean() < 0.35
    xc, yc, w, h = mask_to_box(m)
    assert abs(xc - 0.5) < 0.05 and abs(yc - 0.5) < 0.05 and 0.55 < w < 0.75 and 0.4 < h < 0.6
    assert mask_to_box(np.zeros((10, 10), bool)) is None


def test_health_score_drops_with_lesions():
    healthy = health_indices(fake_leaf(), leaf_mask(fake_leaf()))
    sick_img = fake_leaf(spots=True)
    sick = health_indices(sick_img, leaf_mask(sick_img))
    assert healthy["health_score"] > 95
    assert sick["health_score"] < healthy["health_score"]
    assert healthy["exg"] > 0 and healthy["vari"] > 0


# ---------------------------------------------------------------- irrigation
def test_irrigation_hysteresis_and_safety_cutoff():
    c = IrrigationController(max_on_seconds=100)
    assert c.step(0, 30, 25, 60) is True        # dry -> on
    assert c.step(10, 50, 25, 60) is True       # between thresholds -> stays on
    assert c.step(20, 65, 25, 60) is False      # wet -> off
    assert c.step(30, 50, 25, 60) is False      # between thresholds -> stays off
    assert c.step(40, 10, 25, 60) is True
    assert c.step(141, 20, 25, 60) is False     # safety cut-off after max_on_seconds
    assert c.effective_on_threshold(35, 30) == 40.0


def test_moisture_and_risk():
    assert adc_to_moisture(3200) == 0 and adc_to_moisture(1300) == 100
    assert disease_risk(20, 90) == "high" and disease_risk(20, 75) == "medium" and disease_risk(30, 40) == "low"


# ---------------------------------------------------------------- advisory, inference, API
def test_advisory_covers_all_classes():
    adv = json.loads((ROOT / "app" / "advisory.json").read_text())
    assert set(adv) == set(CLASSES)
    for v in adv.values():
        assert {"disease", "fertilizer", "pesticide", "precautions"} <= set(v)


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    p = tmp_path / "tiny.pt"
    torch.save({"model_name": "small_cnn", "state_dict": SmallCNN().state_dict(), "classes": CLASSES,
                "img_size": 128}, p)
    return str(p)


def test_predict_and_gradcam(tiny_checkpoint, tmp_path):
    from src.predict import predict
    ok, buf = cv2.imencode(".jpg", fake_leaf(True))
    cam_path = tmp_path / "cam.png"
    out = predict(buf.tobytes(), classifier=tiny_checkpoint, gradcam_path=str(cam_path))
    assert out["class"] in CLASSES and 0 <= out["confidence"] <= 1
    assert abs(sum(t["prob"] for t in out["top3"])) <= 1.0001
    assert cam_path.exists() and "health_score" in out["health"]


def test_flask_api(tiny_checkpoint, monkeypatch):
    sys.path.insert(0, str(ROOT / "app"))
    import app as app_module
    monkeypatch.setattr(app_module, "CLASSIFIER", tiny_checkpoint)
    client = app_module.app.test_client()
    assert client.get("/api/health").json["status"] == "ok"
    ok, buf = cv2.imencode(".jpg", fake_leaf())
    r = client.post("/predict", data={"image": (io.BytesIO(buf.tobytes()), "leaf.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and r.json["class"] in CLASSES
    import base64
    r2 = client.post("/predict", json={"image": base64.b64encode(buf.tobytes()).decode()})
    assert r2.status_code == 200
    assert client.post("/predict", json={}).status_code == 400
    s = client.post("/api/sensors", json={"moisture": 20, "temperature": 25, "humidity": 90})
    assert s.json == {"pump": True, "disease_risk": "high"}
    assert client.get("/api/sensors").json["pump"] is True
    assert client.get("/").status_code == 200


def test_background_variants_keep_leaf_pixels():
    from src.eval_backgrounds import backgrounds
    X = np.stack([fake_leaf(), fake_leaf(True)])
    seg = X.copy()
    m = leaf_mask(X[0])
    seg[:, ~m] = 0
    v = backgrounds(X, seg, seed=0)
    assert set(v) == {"original", "black", "flat", "texture"}
    for arr in v.values():
        assert arr.shape == X.shape and arr.dtype == np.uint8
        assert np.array_equal(arr[:, m], X[:, m])          # leaf untouched
    assert np.array_equal(v["black"], seg) and not np.array_equal(v["texture"], X)
