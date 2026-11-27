#!/usr/bin/env bash
# Extra seeds for the SFR ablation (YOLO26n-cls with vs without SFR). Runs after the other queues.
set -u
cd "$(dirname "$0")/.."
while [ ! -f logs/extra_done ]; do sleep 60; done
for seed in 1 2; do
  for m in yolo26_cls_sfr yolo26_cls; do
    python3 -m src.train_classifier --model "$m" --epochs 15 --seed "$seed" --tag "_seed$seed" > "logs/cls_${m}_seed$seed.log" 2>&1
  done
done
echo ALL_DONE > logs/seeds_done
