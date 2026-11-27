#!/usr/bin/env bash
# Background-robust variant (trains on background-removed images 50% of the time). Runs after the detectors.
set -u
cd "$(dirname "$0")/.."
while [ ! -f logs/det_done ]; do sleep 60; done
python3 -m src.train_classifier --model yolo26_cls_sfr --epochs 15 --bg-aug 0.5 > logs/cls_yolo26_cls_sfr_bgaug.log 2>&1
echo ALL_DONE > logs/extra_done
