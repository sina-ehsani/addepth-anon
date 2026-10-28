#!/bin/bash
# Retry of patch4 uniform-6 s1 and encode-prune 1e-3 s1 (OOM'd on GPU7 when 4 ran at once).
# Starts each as soon as one of the running s2 runs frees its slot on GPU7.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
P="--patch_size 4 --hidden 128 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0"
PR="--mode routed --layers 8 --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0 --lambda_budget 1e-3"
run() { local log=$1; shift
  echo "[GPU7 $(date +%F\ %H:%M:%S)] START $log"
  CUDA_VISIBLE_DEVICES=7 $PY -u bench_predrnn.py "$@" > "logs/$log.log" 2>&1
  echo "[GPU7 $(date +%F\ %H:%M:%S)] DONE  $log (exit $?)"; }
( until [ -f results/bench_predrnn_patch4_deep6_L6_s2.json ]; do sleep 60; done
  run patch4_deep6_s1 --mode uniform --layers 6 $P --seed 1 --label patch4_deep6 ) &
( until [ -f results/bench_predrnn_patch4_prune8_l1e3_L8_s2.json ]; do sleep 60; done
  run patch4_prune8_l1e3_s1 $PR $P --seed 1 --label patch4_prune8_l1e3 ) &
wait
