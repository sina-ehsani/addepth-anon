#!/bin/bash
# Positional Pareto-curve extension on GPUs 5/6 (one run each — fits the 8h window;
# positional runs are ~4-5.5h, a 2nd sequential run would overrun). Maps the
# low-compute end the budget reaches below the task-loss optimum (mean depth ~1.83).
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
COMMON="--h5 ../train_test_2015_2020.h5 --key Precipitation --epochs 50 --patience 20 --num_workers 8 --amp"

one() {  # gpu lam seed
  local gpu=$1 lam=$2 s=$3 name="pos_lam${2}_s${3}"
  echo "[GPU$gpu $(date +%H:%M:%S)] START $name"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u pilot_weather.py $COMMON --models positional \
    --lambda_budget "$lam" --seed "$s" --tag "_${name}" --out "pilot_w_${name}.json" \
    > "logs/${name}.log" 2>&1
  echo "[GPU$gpu $(date +%H:%M:%S)] DONE  $name (exit $?)"
}

echo "=== launching 2 positional sweep runs on GPUs 5,6 at $(date) ==="
one 5 0.02 0 &
one 6 0.05 0 &
wait
echo "=== POS SWEEP (5,6) DONE $(date) ==="
