#!/bin/bash
# Extra seeds + latency, scheduled around the running uniform-3 jobs.
#  GPU7 (~10GB free; other user's idle memory): wait for TaxiBJ uni3 chain -> latency on the
#        then-idle GPU -> patchify uniform-6 s1,s2 + v1 encode-prune 1e-3 s1,s2 (4 in parallel).
#  Precip uniform-2 (33%; with uniform-3 at 50% this brackets the router's 40%):
#        GPU1 after precip uni3 s2 (pid 4048348) -> s0, s1 ;  GPU6 after 5.1M uni3 s2 -> s2.
# Recipes copied from deep8_gpu0.sh (uniform-6), prune8_gpu1.sh (prune 1e-3), run_uniform3.sh (precip).
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
P="--patch_size 4 --hidden 128 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --gpu 0"
PR="--mode routed --layers 8 --init_schedule 8 8 8 8 8 8 8 8 8 8 --init_margin 3.0 --lambda_budget 1e-3"
PRE2="--mode deep --layers 2 --val_interval 400 --amp --gpu 0"
run() { local gpu=$1 script=$2 log=$3; shift 3
  echo "[GPU$gpu $(date +%F\ %H:%M:%S)] START $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $script "$@" > "logs/$log.log" 2>&1
  echo "[GPU$gpu $(date +%F\ %H:%M:%S)] DONE  $log (exit $?)"; }

( until [ -f results/bench_taxibj_uni3_L3_s2.json ]; do sleep 60; done; sleep 30
  run 7 bench_latency.py latency_all --iters 200 --warmup 30 --bs 1 16 --out results/latency.json
  for s in 1 2; do
    run 7 bench_predrnn.py patch4_deep6_s$s --mode uniform --layers 6 $P --seed $s --label patch4_deep6 &
    run 7 bench_predrnn.py patch4_prune8_l1e3_s$s $PR $P --seed $s --label patch4_prune8_l1e3 &
  done; wait ) &

( while kill -0 4048348 2>/dev/null; do sleep 60; done
  for s in 0 1; do run 1 bench_predrnn_precip.py uni2_precip_s$s $PRE2 --label uni2_7015 --seed $s; done ) &
( until [ -f results/bench_predrnn_uni3_L3_s2.json ]; do sleep 60; done
  run 6 bench_predrnn_precip.py uni2_precip_s2 $PRE2 --label uni2_7015 --seed 2 ) &

wait
echo "=== EXTRA SEEDS + LATENCY DONE $(date) ==="
