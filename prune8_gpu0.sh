#!/bin/bash
# GPU0: when uniform-6 (deep6) finishes, run PRUNE-FROM-8 router points.
#   Warm-start FIRM at uniform-8 (margin 3.0 -> holds uniform-8 accuracy), then a
#   budget brake prunes depth on positions where it is task-irrelevant (old frames).
#   Money question: does it keep ~uniform-8 accuracy (0.01387) at <100% compute?
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until [ -f results/bench_predrnn_patch4_deep6_L6_s0.json ]; do sleep 30; done
sleep 10
C="--patch_size 4 --hidden 128 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
W="--mode routed --layers 8 --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0"
$PY bench_predrnn.py $W --lambda_budget 5e-4 $C --label patch4_prune8_l5e4 > logs/patch4_prune8_l5e4_s0.log 2>&1
$PY bench_predrnn.py $W --lambda_budget 2e-3 $C --label patch4_prune8_l2e3 > logs/patch4_prune8_l2e3_s0.log 2>&1
