#!/bin/bash
# GPU0: decode-routing v2 prune-from-8. Warm-start encode+decode firm at 8, budget prunes both.
#   l0  : CONTROL -- lambda=0 must reproduce uniform-8 (0.01387); validates decode STE trains clean.
#   l1e3: mid brake.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
W="--mode routed --route_decode --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0"
$PY bench_predrnn.py $W --lambda_budget 0.0  $C --label patch4d_prune8_l0   > logs/patch4d_prune8_l0_s0.log 2>&1
$PY bench_predrnn.py $W --lambda_budget 1e-3 $C --label patch4d_prune8_l1e3 > logs/patch4d_prune8_l1e3_s0.log 2>&1
