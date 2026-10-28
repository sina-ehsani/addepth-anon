#!/bin/bash
# args: $1=GPU  $2=lambda  $3=init_margin  $4=tau_start  $5=label
# START-FROM-6 (bidirectional): encode+decode init at the saturation sweet spot (depth 6),
# small margin so positions can move UP (recent, task-driven) or DOWN (old, budget-driven).
# Light budget brake holds downward tension so it settles NON-UNIFORM instead of growing to 8.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$1
LAM=$2; MARGIN=$3; TAU=$4; LABEL=$5
C="--patch_size 4 --hidden 128 --layers 8 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
$PY bench_predrnn.py --mode routed --route_decode \
  --init_schedule 6 6 6 6 6 6 6 6 6 6 --decode_init 6 6 6 6 6 6 6 6 6 6 --init_margin $MARGIN \
  --tau_start $TAU --tau_end 0.1 --lambda_budget $LAM \
  $C --label $LABEL > logs/${LABEL}_s0.log 2>&1
