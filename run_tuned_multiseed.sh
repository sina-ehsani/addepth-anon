#!/bin/bash
# Completes the TUNED-recipe multi-seed for the 20.36M competitive-scale MMNIST grid.
# Recipe = OpenSTL cheap levers: lr 5e-4 + OneCycleLR + grad_clip 1.0 + 100ep + no-early-stop.
# Each run ~25-32h (100ep, hidden128, full 64x64, no patchify).
#
# ALREADY DONE (do NOT re-run): deepcyc s0,s1 ; initreccyc s0 ; rampcyc s0.
#   -> initreccyc s0 = 0.01431@76% (router) BEATS deepcyc s0/s1 = 0.01514/0.01708 (~11% @ lower compute).
#
# PENDING (this script), in priority order:
#   P1  initreccyc (router) s1, s2   <- headline: confirm the beat is not a lucky seed  [ESSENTIAL]
#   P2  deepcyc s2                    <- complete deep 3-seed baseline
#   P3  rampcyc (hand-set) s1, s2     <- hand-set reduced 3-seed
#   P4  fromunicyc s0                 <- control: does from-uniform basin-failure persist under tuned recipe?
#
# USAGE:  bash run_tuned_multiseed.sh <gpu> [gpu ...]
#   weekend, all parallel (~30h):   bash run_tuned_multiseed.sh 0 1 2 3 4 5
#   essential only, 2 GPUs (~30h):  bash run_tuned_multiseed.sh 0 1     # runs just P1 (router s1,s2); comment out the rest below
#   chain on 2 GPUs, leave 4 free:  bash run_tuned_multiseed.sh 2 3     # 6 runs -> 3 per GPU chained -> ~90h
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
T="--hidden 128 --layers 4 --t_in 10 --t_out 10 --lr 5e-4 --sched onecycle --grad_clip 1.0 --epochs 100 --patience 9999 --val_interval 200 --batch_size 16 --num_workers 8 --gpu 0"
R="--mode routed --init_schedule 1 1 1 1 1 1 1 1 1 4 --init_margin 3.0 --lambda_budget 0.0"
RAMP="--mode custom --encode_depths 1 1 1 2 2 2 3 3 4 4"

# each entry: "argstring|logfile"   (comment out any you don't want)
RUNS=(
  "$R $T --label initreccyc_h128 --seed 1|logs/cyc_initrec_s1.log"
  "$R $T --label initreccyc_h128 --seed 2|logs/cyc_initrec_s2.log"
  "--mode uniform $T --label deepcyc_h128 --seed 2|logs/cyc_deep_s2.log"
  "$RAMP $T --label rampcyc_h128 --seed 1|logs/cyc_ramp_s1.log"
  "$RAMP $T --label rampcyc_h128 --seed 2|logs/cyc_ramp_s2.log"
  "--mode routed --lambda_budget 0.0 $T --label fromunicyc_h128 --seed 0|logs/cyc_fromuni_s0.log"
)

GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && { echo "Usage: bash $0 <gpu> [gpu ...]"; exit 1; }

# round-robin runs onto GPUs; multiple runs on one GPU are CHAINED (sequential) in one tmux session
declare -A CHAIN
for i in "${!RUNS[@]}"; do
  IFS='|' read -r args logf <<< "${RUNS[$i]}"
  g=${GPUS[$(( i % ${#GPUS[@]} ))]}
  CHAIN[$g]+="CUDA_VISIBLE_DEVICES=$g $PY bench_predrnn.py $args > $logf 2>&1"$'\n'
done
for g in "${!CHAIN[@]}"; do
  tmux new-session -d -s cycms_gpu$g "${CHAIN[$g]}"
  echo "== GPU$g =="; echo "${CHAIN[$g]}" | grep -oE "label [a-z0-9_]+ --seed [0-9]"
done
echo "launched. monitor: tmux ls | grep cycms ; results in results/bench_predrnn_*cyc_h128_L4_s*.json"
