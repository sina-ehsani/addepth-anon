#!/bin/bash
# args: $1=GPU  $2=lambda_budget  $3=label
# v2 prune-from-8 with the GOOD recipe from the diagnostic: firm margin 10 + low tau 0.5
# (keeps the clean 0.01353 optimum, avoids the autoregressive stochastic-depth penalty),
# budget brake prunes encode+decode on task-irrelevant positions.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$1
LAM=$2; LABEL=$3
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
$PY bench_predrnn.py --mode routed --route_decode \
  --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 10.0 \
  --tau_start 0.5 --tau_end 0.1 --lambda_budget $LAM \
  $C --label $LABEL > logs/${LABEL}_s0.log 2>&1
