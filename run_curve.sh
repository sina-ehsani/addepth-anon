#!/bin/bash
# Accuracy-vs-compute sweep + fixed-recency control.
#  - routed (HARD straight-through Gumbel, warmup=0, tau 2.0->0.1) over
#    lambda_budget in {0, 0.001, 0.005, 0.02, 0.05} x seeds {0,1,2}  = 15 runs
#  - fixed_schedule [2,2,3,3,5,6] x seeds {0,1,2}                    = 3 runs
# ONE process per GPU (workload is host-mem-bandwidth bound; packing hurts).
# Round-robin across GPUs 0/4/5/6; each lane runs its share sequentially.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
COMMON="--h5 ../train_test_2015_2020.h5 --key Precipitation --epochs 50 --patience 20 --num_workers 8 --amp"
GPUS=(0 4 5 6)

# Build run list. Fields are TAB-delimited: <extra args>\t<tag>\t<name>
specs=()
for lam in 0 0.001 0.005 0.02 0.05; do
  for s in 0 1 2; do
    specs+=("--models routed --router_hard --warmup 0 --tau_start 2.0 --tau_end 0.1 --entropy_weight 0.01 --lambda_budget ${lam} --seed ${s}"$'\t'"_routed_hard_lam${lam}_s${s}"$'\t'"curve_lam${lam}_s${s}")
  done
done
for s in 0 1 2; do
  specs+=("--models fixed_schedule --fixed_schedule 2 2 3 3 5 6 --seed ${s}"$'\t'"_fixedsched_s${s}"$'\t'"fixedsched_s${s}")
done

run_lane() {  # gpu start_index
  local gpu=$1 idx=$2
  while [ "$idx" -lt "${#specs[@]}" ]; do
    IFS=$'\t' read -r margs tag name <<< "${specs[$idx]}"
    echo "[GPU$gpu $(date +%H:%M:%S)] START $name"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u pilot_weather.py $COMMON $margs \
      --tag "$tag" --out "pilot_w_${name}.json" > "logs/${name}.log" 2>&1
    echo "[GPU$gpu $(date +%H:%M:%S)] DONE  $name (exit $?)"
    idx=$((idx + ${#GPUS[@]}))
  done
}

echo "=== launching ${#specs[@]} runs over ${#GPUS[@]} GPUs at $(date) ==="
for i in "${!GPUS[@]}"; do
  run_lane "${GPUS[$i]}" "$i" &
done
wait
echo "=== ALL RUNS DONE $(date) ==="
$PY plot_compute_curve.py
