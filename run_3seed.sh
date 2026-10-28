#!/bin/bash
# 3-seed top-cluster runs (seeds 1 & 2; seed 0 reused from earlier full pilot).
# Models: bidepth, deep, rl, routed(warmup=0). Packed 2 runs per GPU on 0/4/5/6.
# Each run: own --tag (checkpoint isolation), own --out JSON, own unified log.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
COMMON="--h5 ../train_test_2015_2020.h5 --key Precipitation --epochs 50 --patience 20 --num_workers 8 --amp"

run() {  # gpu label "modelargs" seed
  local gpu=$1 label=$2 margs=$3 seed=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY -u pilot_weather.py $COMMON $margs --seed "$seed" \
    --tag "_${label}_s${seed}" --out "pilot_w_seed_${label}_s${seed}.json" \
    > "logs/seed_${label}_s${seed}.log" 2>&1 &
  echo "launched ${label} seed${seed} on GPU${gpu} (pid $!)"
}

# 2 per GPU
run 0 bidepth   "--models bidepth"            1
run 0 deep      "--models deep"               1
run 4 rl        "--models rl"                 1
run 4 routed_w0 "--models routed --warmup 0"  1
run 5 bidepth   "--models bidepth"            2
run 5 deep      "--models deep"               2
run 6 rl        "--models rl"                 2
run 6 routed_w0 "--models routed --warmup 0"  2

echo "=== 8 runs launched at $(date) ==="
wait
echo "=== ALL 8 RUNS DONE at $(date) ==="
$PY aggregate_seeds.py
