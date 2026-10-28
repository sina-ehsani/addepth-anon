#!/bin/bash
# Uniform depth-2 on NYC Taxi (pooled recipe): (14+2)/48 = 33% compute = the router's 33%.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
TAXI="--mode deep --layers 2 --pool_resplit --hidden 64 --window 7 --epochs 80 --patience 20 --val_interval 30 --batch_size 16 --gpu 0"
for s in 0 1 2; do
  ( echo "[GPU6 $(date +%F\ %H:%M:%S)] START uni2_taxi_s$s"
    CUDA_VISIBLE_DEVICES=6 $PY -u bench_predrnn_taxi.py $TAXI --label uni2_pooled --seed $s > logs/uni2_taxi_s$s.log 2>&1
    echo "[GPU6 $(date +%F\ %H:%M:%S)] DONE  uni2_taxi_s$s (exit $?)" ) &
done
wait
echo "=== TAXI UNI2 DONE $(date) ==="
