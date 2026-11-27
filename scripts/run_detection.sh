#!/usr/bin/env bash
# Trains every detector reported in REPORT.md. CPU budget: 35% of the train split, 15 epochs, 256 px (~1 h each).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
while [ ! -f logs/cls_done ]; do sleep 60; done   # wait for the classification queue (shares the 4 CPU cores)
for spec in "yolo26n --sfr" "yolo26n" "yolov5nu" "yolo11n"; do
  tag=$(echo "$spec" | tr -d ' -')
  python3 -m src.train_detector --arch $spec --epochs 15 --fraction 0.35 > "logs/det_${tag}.log" 2>&1
done
echo ALL_DONE > logs/det_done
