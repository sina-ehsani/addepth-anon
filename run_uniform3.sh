#!/bin/bash
# Uniform depth-3 control for Tables 1 & 2 (PredRNN only; ConvLSTM pilot rows skipped).
# Question: is the reduced/router win due to RECENCY allocation, or just less depth?
# A true L=3 model (encode AND decode at depth 3), same recipe as each row's full-depth baseline.
#
# Inference compute vs the full-depth model of that row (JSON compute_frac reads 1.0, i.e.
# relative to its own L=3 -- use these instead):
#   MMNIST 5.1M / 20M (L4, 10->10): (30+30)/80 = 75%   [router 77%, ramp 79%]
#   Precip (L6, 6 stacks -> 1):      (18+3)/42  = 50%   [router 40%]
#   NYC Taxi (L6, 7->1):             (21+3)/48  = 50%   [router 33%]
#   TaxiBJ (L4, 4->4):               (12+12)/32 = 75%   [router 81%]
#
# Precip/TaxiBJ deep launch commands were never saved; these use script defaults
# (+ val_interval 400 for precip, inferred from its log; --amp for both).
#
# Layout: GPUs 0/1/6, two lanes each (one 20M run ~1 day + a chain of small runs).
#   bash run_uniform3.sh   (launch with nohup; logs in logs/uni3_*.log)
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}

M5="--mode uniform --layers 3 --hidden 64 --epochs 50 --patience 15 --amp --num_workers 8 --gpu 0"
M20="--mode uniform --layers 3 --hidden 128 --t_in 10 --t_out 10 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --batch_size 16 --num_workers 8 --gpu 0"
PRE="--mode deep --layers 3 --val_interval 400 --amp --gpu 0"
TAXI="--mode deep --layers 3 --pool_resplit --hidden 64 --window 7 --epochs 80 --patience 20 --val_interval 30 --batch_size 16 --gpu 0"
TBJ="--mode deep --layers 3 --amp --gpu 0"

run() { local gpu=$1 script=$2 log=$3; shift 3
  echo "[GPU$gpu $(date +%F\ %H:%M:%S)] START $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $script "$@" > "logs/$log.log" 2>&1
  echo "[GPU$gpu $(date +%F\ %H:%M:%S)] DONE  $log (exit $?)"; }

# lane A: 20M tuned MMNIST, one seed per GPU
run 0 bench_predrnn.py uni3_20M_s0 $M20 --label uni3cyc_h128 --seed 0 &
run 1 bench_predrnn.py uni3_20M_s1 $M20 --label uni3cyc_h128 --seed 1 &
run 6 bench_predrnn.py uni3_20M_s2 $M20 --label uni3cyc_h128 --seed 2 &

# lane B: smaller runs, chained
( run 0 bench_predrnn_precip.py uni3_precip_s0 $PRE --label uni3_7015 --seed 0
  run 0 bench_predrnn_precip.py uni3_precip_s1 $PRE --label uni3_7015 --seed 1 ) &
( run 1 bench_predrnn_precip.py uni3_precip_s2 $PRE --label uni3_7015 --seed 2
  run 1 bench_predrnn_taxibj.py uni3_taxibj_s0 $TBJ --label uni3 --seed 0 ) &
( for s in 0 1 2; do run 6 bench_predrnn.py uni3_5M_s$s $M5 --label uni3 --seed $s; done
  for s in 1 2; do run 6 bench_predrnn_taxibj.py uni3_taxibj_s$s $TBJ --label uni3 --seed $s; done
  for s in 0 1 2; do run 6 bench_predrnn_taxi.py uni3_taxi_s$s $TAXI --label uni3_pooled --seed $s; done ) &

wait
echo "=== UNIFORM-3 CONTROL DONE $(date) ==="
