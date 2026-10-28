#!/bin/bash
# GPU6: user reprioritized away lambda=3e-3. When lambda=1e-3 finishes, stop the
# sweep6 chain before it launches lambda=3e-3, freeing GPU6 (which lacks the
# memory for the L8 deep experiment anyway -- that runs on GPU0/GPU1).
cd "$(dirname "$0")"
until [ -f results/bench_predrnn_patch4_router_l1e3_L4_s0.json ]; do sleep 30; done
sleep 5
# kill the chain (only lambda=3e-3 remains to run) + any l3e3 proc that just spawned
tmux kill-session -t sweep6 2>/dev/null
pkill -f "patch4_router_l3e3" 2>/dev/null
echo "sweep6 stopped; lambda=3e-3 dropped per reprioritization" > logs/gpu6_dropped.log
