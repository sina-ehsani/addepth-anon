#!/bin/bash
# Seeds 1,2 for the pooled NYC Taxi full-depth and router rows (seed 0 from run_taxi_pooled.sh).
# Same recipe as run_taxi_pooled.sh; resolves the Table 2 (pooled, 1 seed) vs Table 6 (unpooled, 3 seeds) gap.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
C="--pool_resplit --hidden 64 --layers 6 --window 7 --epochs 80 --patience 20 --val_interval 30 --batch_size 16 --gpu 0"
R="--mode routed --init_schedule 1 1 1 1 1 2 3 --init_margin 3.0 --lambda_budget 0.0"
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=0 $PY -u bench_predrnn_taxi.py --mode deep $C --seed $s --label deep_pooled > logs/taxi_deep_pooled_s$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES=6 $PY -u bench_predrnn_taxi.py $R $C --seed $s --label initshallow_pooled > logs/taxi_initshallow_pooled_s$s.log 2>&1 &
done
wait
echo "=== TAXI POOLED SEEDS DONE $(date) ==="
