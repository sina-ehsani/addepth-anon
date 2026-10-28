#!/bin/bash
# GPU6 (headroom card): lambda-frontier, single seed each -> trace accuracy-compute curve
#   from-uniform lambda=0  (high-compute point, grows toward deep)
#   warm-start lambda=1e-3  (moderate brake -> shallower)
#   warm-start lambda=3e-3  (strong brake -> shallowest)
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=6
C="--patch_size 4 --hidden 128 --layers 4 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0"
W="--init_schedule 1 1 1 1 1 1 1 1 1 4 --init_margin 3.0"
$PY bench_predrnn.py --mode routed --lambda_budget 0.0 $C --seed 0 --label patch4_fromuni > logs/patch4_fromuni_l0_s0.log 2>&1
$PY bench_predrnn.py --mode routed $W --lambda_budget 1e-3 $C --seed 0 --label patch4_router_l1e3 > logs/patch4_router_l1e3_s0.log 2>&1
$PY bench_predrnn.py --mode routed $W --lambda_budget 3e-3 $C --seed 0 --label patch4_router_l3e3 > logs/patch4_router_l3e3_s0.log 2>&1
