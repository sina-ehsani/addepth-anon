#!/bin/bash
# Positional GATE (HANDOFF2 tasks 2 + 4). Restricted to GPUs 0 and 4.
# WEATHER positional: lambda in {0 (ceiling), 0.001} x seeds {0,1,2} = 6 runs split
#   across GPU0/GPU4 (full grid, ~4-6h each; no packing — memory-bandwidth bound).
# TAXI positional: lambda=0 x seeds {0,1,2} = 3 runs, on GPU0 FIRST (fast/small), so
#   lambda=0 means it must "discover shallow" from the task loss, not be forced shallow.
# Success(weather): test ~0.463 (ties deep-6, beats routed 0.4676) + recency ramp.
# Success(taxi):    shallow-everywhere schedule -> same method, opposite answer, no tuning.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
WCOMMON="--h5 ../train_test_2015_2020.h5 --key Precipitation --epochs 50 --patience 20 --num_workers 8 --amp"

weather_run() {  # gpu lam seed
  local gpu=$1 lam=$2 s=$3 name="pos_lam${2}_s${3}"
  echo "[W GPU$gpu $(date +%H:%M:%S)] START $name"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u pilot_weather.py $WCOMMON --models positional \
    --lambda_budget "$lam" --seed "$s" --tag "_${name}" --out "pilot_w_${name}.json" \
    > "logs/${name}.log" 2>&1
  echo "[W GPU$gpu $(date +%H:%M:%S)] DONE  $name (exit $?)"
}
taxi_run() {  # seed
  local s=$1
  echo "[T GPU0 $(date +%H:%M:%S)] START taxi_pos_s$s"
  CUDA_VISIBLE_DEVICES=0 $PY -u pilot_taxi.py --models positional --lambda_budget 0 \
    --seed "$s" --out "pilot_taxi_pos_s${s}.json" > "logs/taxi_pos_s${s}.log" 2>&1
  echo "[T GPU0 $(date +%H:%M:%S)] DONE  taxi_pos_s$s (exit $?)"
}

lane_gpu0() {            # 3 fast taxi runs, then half of weather
  for s in 0 1 2; do taxi_run "$s"; done
  weather_run 0 0     0
  weather_run 0 0     2
  weather_run 0 0.001 1
}
lane_gpu4() {            # the other half of weather
  weather_run 4 0     1
  weather_run 4 0.001 0
  weather_run 4 0.001 2
}

echo "=== launching 6 weather + 3 taxi positional runs on GPUs 0,4 at $(date) ==="
lane_gpu0 &
lane_gpu4 &
wait
echo "=== ALL POSITIONAL GATE RUNS DONE $(date) ==="
echo "--- WEATHER positional (refs: deep-6 0.4613 | fixed_sched 0.4630 | routed 0.4676) ---"
$PY - <<'PYEOF'
import json, glob, re
for f in sorted(glob.glob("pilot_w_pos_lam*_s*.json")):
    r=json.load(open(f))["positional"]; lam,s=re.search(r"lam([0-9.]+)_s(\d+)",f).groups()
    print(f"  lam={lam:<6} s{s}  test={r['test_mse']:.4f}  val={r['best_val']:.4f}  mean_depth={r.get('mean_depth')}  depth={[round(x,1) for x in r['depth_profile']]}")
print("--- TAXI positional (expect shallow-everywhere) ---")
for f in sorted(glob.glob("pilot_taxi_pos_s*.json")):
    r=json.load(open(f))["positional"]
    print(f"  {f}: test={r['test_mse']:.4f}  mean_depth={r.get('mean_depth')}  depth={[round(x,1) for x in r['depth_profile']]}")
PYEOF
