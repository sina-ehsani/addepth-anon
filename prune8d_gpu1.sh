#!/bin/bash
# GPU1: decode-routing v2 prune-from-8, lighter + heavier brake to bracket the frontier.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=1
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
W="--mode routed --route_decode --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0"
$PY bench_predrnn.py $W --lambda_budget 5e-4 $C --label patch4d_prune8_l5e4 > logs/patch4d_prune8_l5e4_s0.log 2>&1
$PY bench_predrnn.py $W --lambda_budget 2e-3 $C --label patch4d_prune8_l2e3 > logs/patch4d_prune8_l2e3_s0.log 2>&1
