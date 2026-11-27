#!/usr/bin/env python3
"""Flask REST API + web interface.

Endpoints
  GET  /                 web interface (upload a leaf photo, see diagnosis + advisory + sensors)
  GET  /api/health       liveness check
  POST /predict          multipart form field "image", or JSON {"image": "<base64 jpeg>"} (ESP32-CAM)
  POST /api/sensors      JSON {"moisture": %, "temperature": C, "humidity": %} from the NodeMCU
                         -> returns {"pump": true/false, "disease_risk": ...}
  GET  /api/sensors      latest sensor reading + pump state

    python app/app.py            # http://localhost:5000
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.irrigation import IrrigationController, disease_risk  # noqa: E402
from src.predict import DEFAULT_CLASSIFIER, predict  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
CLASSIFIER = os.environ.get("CLASSIFIER", str(DEFAULT_CLASSIFIER))
DETECTOR = os.environ.get("DETECTOR")  # optional path to a YOLO detector checkpoint
controller = IrrigationController()
latest_sensors: dict = {}


def _image_bytes() -> bytes | None:
    if "image" in request.files:
        return request.files["image"].read()
    data = request.get_json(silent=True) or {}
    b64 = data.get("image")
    if b64:
        if "," in b64[:100]:  # strip a data-URL prefix
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)
    return None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "classifier": Path(CLASSIFIER).name, "detector": DETECTOR})


@app.post("/predict")
def predict_route():
    raw = _image_bytes()
    if not raw:
        return jsonify({"error": "send an image as multipart field 'image' or JSON {'image': base64}"}), 400
    try:
        t0 = time.time()
        res = predict(raw, classifier=CLASSIFIER, detector=DETECTOR)
        res["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/sensors")
def sensors_post():
    d = request.get_json(force=True)
    try:
        m, t, h = float(d["moisture"]), float(d["temperature"]), float(d["humidity"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "need numeric moisture, temperature, humidity"}), 400
    pump = controller.step(time.time(), m, t, h)
    latest_sensors.update({"moisture": m, "temperature": t, "humidity": h, "pump": pump,
                           "disease_risk": disease_risk(t, h), "time": time.time()})
    return jsonify({"pump": pump, "disease_risk": latest_sensors["disease_risk"]})


@app.get("/api/sensors")
def sensors_get():
    return jsonify(latest_sensors or {"status": "no readings yet"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
