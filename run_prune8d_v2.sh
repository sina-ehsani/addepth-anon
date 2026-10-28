#!/bin/bash
# args: $1=GPU  $2=lambda  $3=init_margin  $4=tau_start  $5=label
# Decouple the tension: MARGIN sets starting logit gap (movability), TAU sets sampling noise.
# v1 showed margin 3 + lambda 2e-3 PRUNES; diagnostic showed low tau 0.5 removes the
# autoregressive rollout penalty. Combine: movable margin + clean low-tau sampling.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$1
LAM=$2; MARGIN=$3; TAU=$4; LABEL=$5
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
$PY bench_predrnn.py --mode routed --route_decode \
  --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin $MARGIN \
  --tau_start $TAU --tau_end 0.1 --lambda_budget $LAM \
  $C --label $LABEL > logs/${LABEL}_s0.log 2>&1
