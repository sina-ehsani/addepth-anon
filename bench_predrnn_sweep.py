"""HANDOFF3 Phase-0, step 2 (FREE): is PredRNN's depth-need uneven across positions?

Loads the trained deep (L4) PredRNN and, WITHOUT retraining, evaluates it under
many per-position encode schedules (set_encode_depths). Shows which positions
tolerate depth-skipping -> reveals PredRNN's own sensitivity profile, rather than
assuming the ConvLSTM recency schedule. Conservative test: the model was trained
full-depth, so a schedule that survives here is a strong GO (a retrained-reduced
model can only do better). Decode stays full-depth (v1).
"""
import json, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from bench_predrnn import MM10
from src.models.PredRNN import VarDepthPredRNN
from src.train_loop import evaluate_mse

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
arr = np.transpose(np.load("../mnist_test_seq.npy"), (1, 0, 2, 3)).astype(np.float32) / 255.0
te = DataLoader(Subset(MM10(arr, 10, 10), range(9000, 10000)), batch_size=16, shuffle=False,
                num_workers=8, pin_memory=True)

m = VarDepthPredRNN(1, 64, 4, t_out=10, ksize=5, tensor_shape=(64, 64)).to(dev)
m.load_state_dict(torch.load("checkpoints/predrnn_uniform_L4.pt", weights_only=True))

SCHEDULES = {
    "full        [4x10]":            [4] * 10,
    "recency     [1x9,4]":           [1] * 9 + [4],
    "last2-deep  [1x8,4,4]":         [1] * 8 + [4, 4],
    "last-half   [1x5,4x5]":         [1] * 5 + [4] * 5,
    "ramp        [1,1,1,2,2,2,3,3,4,4]": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    "anti-recency[4,1x9]":           [4] + [1] * 9,
    "all-2       [2x10]":            [2] * 10,
    "all-3       [3x10]":            [3] * 10,
}
DEEP = 0.02308      # trained deep baseline (per-pixel)
print(f"PredRNN L4 encode-schedule sweep (inference-ablation of trained deep; baseline {DEEP:.5f})")
print(f"  {'schedule':32s} {'test MSE':>9s} {'vs deep':>9s} {'enc-compute':>11s}")
rows = []
for name, sch in SCHEDULES.items():
    m.set_encode_depths(sch)
    mse = evaluate_mse(m, te, dev)
    comp = 100 * m.encode_compute_frac(10)
    rows.append((mse, name, comp))
for mse, name, comp in rows:
    gap = 100 * (mse - DEEP) / DEEP
    print(f"  {name:32s} {mse:9.5f} {gap:+8.1f}% {comp:10.0f}%")
print("\nGO if some reduced schedule stays near deep (small gap) at <100% compute; "
      "if all degrade toward shallow, depth-need is ~uniform (routing won't save).")
