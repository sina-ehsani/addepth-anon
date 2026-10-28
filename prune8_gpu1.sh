#!/bin/bash
# GPU1: when router-8 from-uniform (r8fromuni) finishes, run the middle prune point.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=1
until [ -f results/bench_predrnn_patch4_r8fromuni_L8_s0.json ]; do sleep 30; done
sleep 10
C="--patch_size 4 --hidden 128 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
W="--mode routed --layers 8 --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0"
$PY bench_predrnn.py $W --lambda_budget 1e-3 $C --label patch4_prune8_l1e3 > logs/patch4_prune8_l1e3_s0.log 2>&1
