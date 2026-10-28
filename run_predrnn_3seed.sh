#!/bin/bash
# PredRNN GO confirmation: 3-seed deep/ramp/recency + a long-trained deep.
#  - reuse seed-0 (already computed) -> seeded names
#  - run seeds 1,2 for deep/ramp/recency (standard recipe: epochs 50, patience 15)
#  - deep_long (epochs 150, patience 40) to rule out "deep baseline under-optimized"
# GPUs 0,1,3,7. Confirm: does reduced (ramp/recency) BEAT deep robustly across seeds,
# and does a longer-trained deep close the gap?
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
STD="--layers 4 --hidden 64 --epochs 50 --patience 15 --amp --num_workers 8 --gpu 0"
RAMP="1 1 1 2 2 2 3 3 4 4"; REC="1 1 1 1 1 1 1 1 1 4"

# reuse seed-0 results (valid, already trained) under the seeded naming
cp -n results/bench_predrnn_uniform_L4.json results/bench_predrnn_deep_L4_s0.json 2>/dev/null || true
cp -n results/bench_predrnn_ramp_L4.json    results/bench_predrnn_ramp_L4_s0.json 2>/dev/null || true
cp -n results/bench_predrnn_recency_L4.json results/bench_predrnn_recency_L4_s0.json 2>/dev/null || true

run() { local gpu=$1 log=$2; shift 2
  echo "[GPU$gpu $(date +%H:%M:%S)] START $log"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u bench_predrnn.py "$@" > "logs/$log.log" 2>&1
  echo "[GPU$gpu $(date +%H:%M:%S)] DONE  $log"; }

( run 0 deep_s1    --mode uniform --label deep    --seed 1 $STD
  run 0 recency_s1 --mode custom --encode_depths $REC  --label recency --seed 1 $STD ) &
( run 1 deep_s2    --mode uniform --label deep    --seed 2 $STD
  run 1 recency_s2 --mode custom --encode_depths $REC  --label recency --seed 2 $STD ) &
( run 3 ramp_s1    --mode custom --encode_depths $RAMP --label ramp    --seed 1 $STD
  run 3 ramp_s2    --mode custom --encode_depths $RAMP --label ramp    --seed 2 $STD ) &
( run 7 deep_long  --mode uniform --label deep_long --seed 0 \
    --layers 4 --hidden 64 --epochs 150 --patience 40 --amp --num_workers 8 --gpu 0 ) &
wait
echo "=== PREDRNN 3-SEED + DEEP_LONG DONE $(date) ==="
$PY - <<'PYEOF'
import json, glob, statistics as st
def vals(lab):
    xs=[]
    for f in sorted(glob.glob(f"results/bench_predrnn_{lab}_L4_s*.json")):
        xs.append(json.load(open(f))["test_mse_perpixel"])
    return xs
print("PredRNN 3-seed test MSE (per-pixel):")
for lab in ["deep","ramp","recency"]:
    xs=vals(lab); m=st.mean(xs); s=st.pstdev(xs) if len(xs)>1 else 0
    print(f"  {lab:8s} {m:.5f} +/- {s:.5f}  n={len(xs)}  {[round(x,5) for x in xs]}")
try:
    dl=json.load(open("results/bench_predrnn_deep_long_L4_s0.json"))
    print(f"  deep_long(s0, 150ep): {dl['test_mse_perpixel']:.5f}  (vs deep s0 0.02308)")
except FileNotFoundError: pass
dm=st.mean(vals("deep")); rm=st.mean(vals("ramp"))
print(f"\n  ramp mean vs deep mean: {100*(rm-dm)/dm:+.1f}%  -> reduced {'WINS (GO holds)' if rm<dm else 'does NOT win (reconsider)'}")
PYEOF
