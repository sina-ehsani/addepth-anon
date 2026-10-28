#!/bin/bash
# GPU1: when router_s2 (L4) finishes, run max_depth=8 ROUTERS.
#   r8grow  : warm-start ALL positions at depth 4 (uniform-4 prior), free to grow to 8
#             -> directly tests "start at deep, explore deeper if it helps".
#   r8fromuni: no prior, router free to choose in [1..8] from uniform init.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=1
until [ -f results/bench_predrnn_patch4_router_L4_s2.json ]; do sleep 30; done
sleep 10
C="--patch_size 4 --hidden 128 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0 --seed 0 --lambda_budget 0.0"
$PY bench_predrnn.py --mode routed --layers 8 --init_schedule 4 4 4 4 4 4 4 4 4 4 --init_margin 3.0 $C --label patch4_r8grow > logs/patch4_r8grow_s0.log 2>&1
$PY bench_predrnn.py --mode routed --layers 8 $C --label patch4_r8fromuni > logs/patch4_r8fromuni_s0.log 2>&1
