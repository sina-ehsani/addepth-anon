#!/bin/bash
# HANDOFF3 Phase-0 confirmation: TRAIN reduced-encode PredRNN end-to-end (not just
# inference-ablate) to check it still can't reach deep (0.02308). Two schedules:
#   ramp    [1,1,1,2,2,2,3,3,4,4]  -- least-bad in the ablation sweep (+81%)
#   recency [1x9,4]                -- our ConvLSTM method's schedule (+165% ablated)
# If a TRAINED reduced model still trails deep -> NO-GO is airtight.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
COMMON="--mode custom --layers 4 --hidden 64 --epochs 50 --patience 15 --amp --num_workers 8 --gpu 0"

CUDA_VISIBLE_DEVICES=0 $PY -u bench_predrnn.py $COMMON --label ramp \
  --encode_depths 1 1 1 2 2 2 3 3 4 4 > logs/predrnn_confirm_ramp.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u bench_predrnn.py $COMMON --label recency \
  --encode_depths 1 1 1 1 1 1 1 1 1 4 > logs/predrnn_confirm_recency.log 2>&1 &
echo "=== launched ramp(GPU0) + recency(GPU1) at $(date) ==="
wait
echo "=== CONFIRM DONE $(date) ==="
$PY - <<'PYEOF'
import json
print("deep baseline (trained full-depth): 0.02308")
for lab in ["ramp","recency"]:
    try:
        r=json.load(open(f"results/bench_predrnn_{lab}_L4.json"))
        gap=100*(r['test_mse_perpixel']-0.02308)/0.02308
        print(f"  {lab:8s} (trained): {r['test_mse_perpixel']:.5f}  {gap:+.1f}% vs deep  | compute {100*r['compute_frac']:.0f}%")
    except FileNotFoundError:
        print(f"  {lab}: missing")
print("Airtight NO-GO if both TRAINED reduced schedules still trail deep clearly.")
PYEOF
