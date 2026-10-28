"""Regression test for PredRNNRouted (the learned-router PredRNN).

Guards the STE state-handoff — that the routed forward reproduces the discrete
VarDepthPredRNN semantics exactly, and that gradient reaches pos_logits.
  A. internal equivalence: routed eval (pinned schedule) == VarDepthPredRNN(sched)
  B. train == eval: STE keep-gate mixture value == discrete eval value (one-hot)
  C. (optional) end-to-end: load the trained deep checkpoint, pin full depth,
     reproduce the published deep MSE (0.02308).
Run: python test_predrnn_routed.py   (prints ALL PREDRNN-ROUTED CHECKS PASSED)
"""
import os
import torch
from src.models.PredRNN import VarDepthPredRNN, PredRNNRouted

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
H, L, TIN, TOUT = 64, 4, 10, 10
SCHEDS = [[1, 1, 1, 2, 2, 2, 3, 3, 4, 4], [1] * 9 + [4]]


def pin(router, sched, min_depth=1, margin=20.0):
    with torch.no_grad():
        router.pos_logits.zero_()
        for i, d in enumerate(sched):
            router.pos_logits[i, d - min_depth] = margin


def build():
    ref = VarDepthPredRNN(1, H, L, TOUT, ksize=5, tensor_shape=(64, 64)).to(dev)
    rout = PredRNNRouted(1, H, L, TOUT, window=TIN, min_depth=1, ksize=5,
                         tensor_shape=(64, 64), hard_routing=True, positional=True).to(dev)
    rout.load_state_dict(ref.state_dict(), strict=False)   # cells.* / out_conv.* overlap
    return ref, rout


# ---- Guard A: internal equivalence (eval) + gradient to pos_logits ----
ref, rout = build()
X = torch.randn(2, TIN, 1, 64, 64, device=dev)
ref.eval(); rout.eval()
for sched in SCHEDS:
    ref.set_encode_depths(sched); pin(rout.router, sched)
    with torch.no_grad():
        o_ref, o_rout = ref(X), rout(X)
    md = (o_ref - o_rout).abs().max().item()
    assert torch.allclose(o_ref, o_rout, atol=1e-4), f"A: mismatch {sched}: max|Δ|={md}"
    print(f"A ok: routed eval == VarDepthPredRNN {sched}  (max|Δ|={md:.2e})")

rout.train(); pin(rout.router, SCHEDS[0])
rout.zero_grad()
loss = rout(X).pow(2).mean(); loss.backward()
g = rout.router.pos_logits.grad
assert g is not None and g.abs().sum() > 0, "A: no gradient to pos_logits"
nz_rows = (g.abs().sum(-1) > 0).sum().item()
print(f"A ok: pos_logits.grad flows (nonzero rows={nz_rows}/{TIN}, |g|sum={g.abs().sum():.3e})")

# ---- Guard B: train forward value == eval forward value (one-hot) ----
ref, rout = build()
pin(rout.router, SCHEDS[1], margin=1e4)   # huge margin => Gumbel-hard argmax deterministic
rout.train()
with torch.no_grad():
    o_train = rout(X)
rout.eval()
with torch.no_grad():
    o_eval = rout(X)
md = (o_train - o_eval).abs().max().item()
assert torch.allclose(o_train, o_eval, atol=1e-4), f"B: train!=eval max|Δ|={md}"
print(f"B ok: train (STE mixture) == eval (discrete)  (max|Δ|={md:.2e})")

# ---- Guard C (optional): end-to-end vs published deep MSE ----
ckpt = "checkpoints/predrnn_uniform_L4.pt"; npy = "../mnist_test_seq.npy"
if os.path.exists(ckpt) and os.path.exists(npy):
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from bench_predrnn import MM10
    from src.train_loop import evaluate_mse
    arr = np.transpose(np.load(npy), (1, 0, 2, 3)).astype(np.float32) / 255.0
    te = DataLoader(Subset(MM10(arr, TIN, TOUT), range(9000, 10000)), batch_size=16,
                    shuffle=False, num_workers=8, pin_memory=True)
    m = PredRNNRouted(1, H, L, TOUT, window=TIN, min_depth=1, ksize=5, tensor_shape=(64, 64)).to(dev)
    m.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
    pin(m.router, [L] * TIN)                # full depth => should reproduce deep 0.02308
    mse = evaluate_mse(m, te, dev)
    print(f"C: deep-checkpoint + full-depth routing -> test MSE {mse:.5f} (published deep 0.02308)")
    assert abs(mse - 0.02308) < 5e-4, f"C: {mse} not ~0.02308"
    print("C ok: reproduces published deep MSE")
else:
    print("C skipped (checkpoint or npy not present)")

print("\nALL PREDRNN-ROUTED CHECKS PASSED")
