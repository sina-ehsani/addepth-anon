#!/bin/bash
# GPU0: deep (uniform) seed replicates s1, s2  -> headline error bars
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
C="--patch_size 4 --hidden 128 --layers 4 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0"
$PY bench_predrnn.py --mode uniform $C --seed 1 --label patch4_deep > logs/patch4_deep_s1.log 2>&1
$PY bench_predrnn.py --mode uniform $C --seed 2 --label patch4_deep > logs/patch4_deep_s2.log 2>&1
