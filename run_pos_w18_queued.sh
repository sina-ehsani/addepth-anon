#!/bin/bash
# Queue: wait for the window=12 run to finish (frees GPU0), then run window=18 s0
# (the decisive comb test: depth teeth expected at pos17~now, ~pos11 24h-ago,
# ~pos5 48h-ago if the schedule is diurnal). lambda=0.001 so depth must pay for itself.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
echo "[$(date)] waiting for window=12 to finish (pilot_w_pos_w12_s0.json)..."
while [ ! -f pilot_w_pos_w12_s0.json ]; do sleep 120; done
echo "[$(date)] window=12 done -> launching window=18 s0 on GPU0"
CUDA_VISIBLE_DEVICES=0 $PY -u pilot_weather.py --h5 ../train_test_2015_2020.h5 --key Precipitation \
  --models positional --window 18 --lambda_budget 0.001 --seed 0 \
  --epochs 50 --patience 20 --num_workers 8 --amp \
  --tag _pos_w18_s0 --out pilot_w_pos_w18_s0.json > logs/pos_w18_s0.log 2>&1
echo "[$(date)] window=18 s0 DONE (exit $?)"
