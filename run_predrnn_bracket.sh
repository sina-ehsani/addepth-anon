#!/bin/bash
# HANDOFF3 Phase-0, step 1: DOES DEPTH MATTER on PredRNN? (the fundamental gate)
# Train deep (4-layer) vs shallow (1-layer) uniform PredRNN on MMNIST 10->10.
#   deep >> shallow  -> depth matters -> proceed to the schedule sweep (step 2)
#   deep ~= shallow  -> depth irrelevant (like taxi) -> stop, clean negative.
# Sequential on GPU 0 (only free GPU). shallow (L1) is cheap; deep (L4) heavier.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
COMMON="--mode uniform --hidden 64 --epochs 50 --patience 15 --amp --num_workers 8 --gpu 0"

echo "=== [GPU0] deep (L4) START $(date) ==="
CUDA_VISIBLE_DEVICES=0 $PY -u bench_predrnn.py $COMMON --layers 4 > logs/predrnn_deep_L4.log 2>&1
echo "=== [GPU0] deep done; shallow (L1) START $(date) ==="
CUDA_VISIBLE_DEVICES=0 $PY -u bench_predrnn.py $COMMON --layers 1 > logs/predrnn_shallow_L1.log 2>&1
echo "=== BRACKET DONE $(date) ==="
$PY - <<'PYEOF'
import json
def get(f):
    try: return json.load(open(f))
    except FileNotFoundError: return None
d=get("results/bench_predrnn_uniform_L4.json"); s=get("results/bench_predrnn_uniform_L1.json")
for name,r in [("deep (L4)",d),("shallow (L1)",s)]:
    if r: print(f"  {name:12s}: per-pixel {r['test_mse_perpixel']:.5f} (OpenSTL {r['test_mse_openstl']:.1f}) | {r['params_M']:.2f}M")
if d and s:
    gap=100*(s['test_mse_perpixel']-d['test_mse_perpixel'])/d['test_mse_perpixel']
    print(f"  shallow vs deep accuracy gap: {gap:+.1f}%")
    print("  -> DEPTH MATTERS (proceed to schedule sweep)" if gap>5 else "  -> depth ~irrelevant (stop; clean negative like taxi)")
PYEOF
