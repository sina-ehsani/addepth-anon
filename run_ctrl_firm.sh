#!/bin/bash
# Diagnostic: is the v2 decode-routing control gap (0.01419 vs uniform-8 0.01387)
# reducible Gumbel-sampling noise, or intrinsic to the routed forward?
# Make routing near-DETERMINISTIC at depth 8: firm margin 10 + low tau_start 0.5.
# If this reproduces ~0.01387 -> gap was stochastic-depth training noise (reducible).
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
$PY bench_predrnn.py --mode routed --route_decode \
  --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 10.0 \
  --tau_start 0.5 --tau_end 0.1 --lambda_budget 0.0 \
  $C --label patch4d_ctrl_firm > logs/patch4d_ctrl_firm_s0.log 2>&1
