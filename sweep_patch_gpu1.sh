#!/bin/bash
# GPU1: warm-start router (lambda=0) seed replicates s1, s2  -> headline error bars
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=1
C="--patch_size 4 --hidden 128 --layers 4 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0"
W="--mode routed --init_schedule 1 1 1 1 1 1 1 1 1 4 --init_margin 3.0 --lambda_budget 0.0"
$PY bench_predrnn.py $W $C --seed 1 --label patch4_router > logs/patch4_router_s1.log 2>&1
$PY bench_predrnn.py $W $C --seed 2 --label patch4_router > logs/patch4_router_s2.log 2>&1
