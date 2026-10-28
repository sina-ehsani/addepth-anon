#!/bin/bash
# Reroute of run_uniform3.sh lane-B chains (GPU7 joined; it only has ~10GB free, so small jobs only).
#   GPU7: TaxiBJ s0-2 chain  ||  NYC Taxi uni3 s0-2 chain
#   GPU6: 5.1M s1, s2 after the running 5.1M s0 (pid 4048351) exits
# Precip s0->s1 (GPU0) and precip s2 (GPU1) are unchanged; 20M runs untouched.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
M5="--mode uniform --layers 3 --hidden 64 --epochs 50 --patience 15 --amp --num_workers 8 --gpu 0"
TAXI="--mode deep --layers 3 --pool_resplit --hidden 64 --window 7 --epochs 80 --patience 20 --val_interval 30 --batch_size 16 --gpu 0"
TBJ="--mode deep --layers 3 --amp --gpu 0"
run() { local gpu=$1 script=$2 log=$3; shift 3
  echo "[GPU$gpu $(date +%F\ %H:%M:%S)] START $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $script "$@" > "logs/$log.log" 2>&1
  echo "[GPU$gpu $(date +%F\ %H:%M:%S)] DONE  $log (exit $?)"; }
( for s in 0 1 2; do run 7 bench_predrnn_taxibj.py uni3_taxibj_s$s $TBJ --label uni3 --seed $s; done ) &
( for s in 0 1 2; do run 7 bench_predrnn_taxi.py uni3_taxi_s$s $TAXI --label uni3_pooled --seed $s; done ) &
( while kill -0 4048351 2>/dev/null; do sleep 60; done
  for s in 1 2; do run 6 bench_predrnn.py uni3_5M_s$s $M5 --label uni3 --seed $s; done ) &
wait
echo "=== REROUTE DONE $(date) ==="
