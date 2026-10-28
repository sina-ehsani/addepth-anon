#!/bin/bash
# HANDOFF3 Phase-0 go/no-go: uniform vs reduced encode-depth PredRNN on MMNIST 10->10.
# Two GPUs, one mode each. Compare the RELATIVE accuracy gap vs compute saved.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
COMMON="--hidden 64 --layers 4 --epochs 50 --patience 15 --amp --num_workers 8"

CUDA_VISIBLE_DEVICES=5 $PY -u bench_predrnn.py --mode uniform $COMMON > logs/predrnn_uniform.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -u bench_predrnn.py --mode reduced $COMMON > logs/predrnn_reduced.log 2>&1 &
echo "=== launched uniform(GPU5) + reduced(GPU6) at $(date) ==="
wait
echo "=== PHASE-0 DONE $(date) ==="
$PY - <<'PYEOF'
import json
for m in ["uniform","reduced"]:
    try:
        r=json.load(open(f"results/bench_predrnn_{m}.json"))
        print(f"  {m:8s}: per-pixel MSE {r['test_mse_perpixel']:.5f} | OpenSTL-conv {r['test_mse_openstl']:.1f} "
              f"| compute {100*r['compute_frac']:.0f}% | depths {r['encode_depths']}")
    except FileNotFoundError:
        print(f"  {m}: missing")
print("GO if reduced ~= uniform accuracy at <100% compute; NO-GO if reduced markedly worse.")
PYEOF
