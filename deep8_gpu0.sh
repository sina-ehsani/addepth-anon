#!/bin/bash
# GPU0: when deep_s2 (L4) finishes, run the deeper-UNIFORM baselines.
#   uniform-8 (new accuracy ceiling) then uniform-6.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until [ -f results/bench_predrnn_patch4_deep_L4_s2.json ]; do sleep 30; done
sleep 10
C="--patch_size 4 --hidden 128 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0"
$PY bench_predrnn.py --mode uniform --layers 8 $C --label patch4_deep8 > logs/patch4_deep8_s0.log 2>&1
$PY bench_predrnn.py --mode uniform --layers 6 $C --label patch4_deep6 > logs/patch4_deep6_s0.log 2>&1
