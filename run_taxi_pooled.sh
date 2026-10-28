#!/bin/bash
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=6
C="--pool_resplit --hidden 64 --layers 6 --window 7 --epochs 80 --patience 20 --val_interval 30 --batch_size 16 --gpu 0 --seed 0"
$PY bench_predrnn_taxi.py --mode deep $C --label deep_pooled > logs/taxi_deep_pooled_s0.log 2>&1
$PY bench_predrnn_taxi.py --mode routed --init_schedule 1 1 1 1 1 2 3 --init_margin 3.0 --lambda_budget 0.0 $C --label initshallow_pooled > logs/taxi_initshallow_pooled_s0.log 2>&1
$PY bench_predrnn_taxi.py --mode routed --lambda_budget 0.0 $C --label fromuniform_pooled > logs/taxi_fromuniform_pooled_s0.log 2>&1
