"""Regression guards for decode-routing v2 (PredRNNRouted route_decode=True).

A: decode pinned to max_depth == v1 full-depth decode (same weights)   [head/state handoff]
B: train forward == eval forward for a fixed one-hot decode schedule    [decode STE correctness]
C: gradient reaches the DECODE pos_logits and varies by decode position [learnability]
"""
import copy, torch
from src.models.PredRNN import PredRNNRouted

torch.manual_seed(0)
IN_CH, HID, MAXD, T_OUT, WIN, MND = 1, 8, 4, 5, 6, 1
SHAPE = (8, 8)
K = MAXD - MND + 1


def build(route_decode):
    return PredRNNRouted(IN_CH, HID, MAXD, T_OUT, window=WIN, min_depth=MND, ksize=3,
                         tensor_shape=SHAPE, hard_routing=True, positional=True,
                         collapse_out=False, route_decode=route_decode)


def pin(model, pos, depth):
    with torch.no_grad():
        model.router.pos_logits[pos].zero_()
        model.router.pos_logits[pos, depth - MND] = 10.0


X = torch.randn(3, WIN, IN_CH, SHAPE[0], SHAPE[1])

# ---- A: decode pinned to max == v1 ----
v1 = build(False).eval()
v2 = build(True).eval()
# share cells + out_conv weights (routers differ in table size)
sd = {k: v for k, v in v1.state_dict().items() if k.startswith(("cells", "out_conv"))}
v2.load_state_dict(sd, strict=False)
for p in range(WIN):                       # encode: full depth on BOTH (v1 encode is routed too)
    pin(v1, p, MAXD)
    pin(v2, p, MAXD)
for t in range(T_OUT):                      # decode: pinned to max
    pin(v2, WIN + t, MAXD)
with torch.no_grad():
    o1, o2 = v1(X), v2(X)
errA = (o1 - o2).abs().max().item()
assert errA < 1e-5, f"A FAILED: decode-pinned-max != v1 (max err {errA:.2e})"
print(f"A PASS: decode pinned to max == v1 full-depth decode (max err {errA:.2e})")

# ---- B: train == eval for a fixed one-hot reduced decode schedule ----
sched_dec = [1, 2, 3, 4, 2]
for p in range(WIN):
    pin(v2, p, 3)                           # some fixed encode schedule
for t, d in enumerate(sched_dec):
    pin(v2, WIN + t, d)
v2.router.set_temperature(0.1)
v2.train()
with torch.no_grad():
    o_train = v2(X)
v2.eval()
with torch.no_grad():
    o_eval = v2(X)
errB = (o_train - o_eval).abs().max().item()
assert errB < 1e-5, f"B FAILED: train != eval decode (max err {errB:.2e})"
print(f"B PASS: train==eval for fixed decode schedule {sched_dec} (max err {errB:.2e})")

# ---- C: gradient reaches decode pos_logits, varies by position ----
v3 = build(True).train()
v3.router.pos_logits.grad = None
v3.router.set_temperature(1.0)
out = v3(X)
loss = out.pow(2).mean()
loss.backward()
g = v3.router.pos_logits.grad
assert g is not None, "C FAILED: no grad on pos_logits"
dec_g = g[WIN:WIN + T_OUT]                   # decode rows
gnorm = dec_g.abs().sum().item()
nonzero_rows = (dec_g.abs().sum(-1) > 1e-9).sum().item()
assert gnorm > 1e-8, f"C FAILED: decode grad ~zero ({gnorm:.2e})"
assert nonzero_rows >= 2, f"C FAILED: decode grad hits only {nonzero_rows} positions"
print(f"C PASS: decode pos_logits grad nonzero on {nonzero_rows}/{T_OUT} positions (|g|={gnorm:.3e})")
print("\nALL DECODE-ROUTING GUARDS PASS")
