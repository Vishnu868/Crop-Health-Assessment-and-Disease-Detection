#!/usr/bin/env bash
# Trains every classifier reported in REPORT.md (real PlantVillage data). ~1-2 h on a 4-core CPU.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
for m in yolo26_cls_sfr yolo26_cls yolo11_cls yolo11_cls_sfr small_cnn; do
  ep=15; [ "$m" = small_cnn ] && ep=20
  python3 -m src.train_classifier --model "$m" --epochs "$ep" > "logs/cls_${m}.log" 2>&1
done
python3 -m src.train_classifier --model yolo26_cls_sfr --epochs 15 --no-pretrained > logs/cls_yolo26_cls_sfr_scratch.log 2>&1
echo ALL_DONE > logs/cls_done
