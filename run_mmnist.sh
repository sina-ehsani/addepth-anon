#!/bin/bash
# Moving MNIST pilot, full model set x 3 seeds. One seed per GPU (2,3,4); MMNIST
# is small/fast (~2-5 min/model). Tests cross-dataset generalization of the
# positional depth schedule against the same baselines/controls as weather.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
MODELS="positional deep shallow convlstm fixed_schedule bidepth routed"
GPUS=(2 3 4)

lane() {  # gpu seed
  local gpu=$1 s=$2
  echo "[GPU$gpu $(date +%H:%M:%S)] START mmnist seed$s"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u pilot_mmnist.py --models $MODELS --seed "$s" \
    --epochs 50 --patience 20 --num_workers 8 --amp \
    --tag "_s${s}" --out "pilot_mm_s${s}.json" > "logs/mmnist_s${s}.log" 2>&1
  echo "[GPU$gpu $(date +%H:%M:%S)] DONE  mmnist seed$s (exit $?)"
}

echo "=== launching mmnist seeds 0,1,2 on GPUs 2,3,4 at $(date) ==="
lane 2 0 &
lane 3 1 &
lane 4 2 &
wait
echo "=== MMNIST DONE $(date) ==="
$PY - <<'PYEOF'
import json, glob
from collections import defaultdict
import statistics as st
by=defaultdict(list)
for f in sorted(glob.glob("pilot_mm_s*.json")):
    d=json.load(open(f))
    for k,v in d.items(): by[k].append(v["test_mse"])
print("Moving MNIST — test MSE (mean over seeds):")
for k in sorted(by, key=lambda k: st.mean(by[k])):
    vals=by[k]; m=st.mean(vals); s=st.pstdev(vals) if len(vals)>1 else 0
    print(f"  {k:22s} {m:.5f} +/- {s:.5f}  (n={len(vals)})")
# positional schedules
print("positional schedules per seed:")
for f in sorted(glob.glob("pilot_mm_s*.json")):
    p=json.load(open(f)).get("positional")
    if p: print(f"  {f}: depth={[round(x,1) for x in p['depth_profile']]} mean={p.get('mean_depth')}")
PYEOF
