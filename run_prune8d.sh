#!/bin/bash
# args: $1=GPU  $2=lambda_budget  $3=label
# decode-routing v2 prune-from-8: warm-start encode+decode firm at 8, budget prunes both.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$1
LAM=$2; LABEL=$3
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
W="--mode routed --route_decode --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0"
$PY bench_predrnn.py $W --lambda_budget $LAM $C --label $LABEL > logs/${LABEL}_s0.log 2>&1
