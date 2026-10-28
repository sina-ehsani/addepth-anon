"""Quick torch-level sanity check for the positional-only depth schedule.

Run on the server (where torch is installed):
    python test_positional.py

Verifies, on tiny random data:
  1. AdDepthRouted(positional=True) forward produces correct output shape.
  2. The router distribution is identical across batch items (input-independent).
  3. pos_logits receives gradients (the schedule is actually learning).
  4. expected_depth / depth_budget / router_entropy all work.
  5. Eval mode produces a one-hot (hard) per-position depth.
"""

import torch
import torch.nn as nn

from src.models.DepthRouter import AdDepthRouted

b, window, ch, H, W = 4, 6, 8, 16, 16
min_depth, max_depth = 1, 6

model = AdDepthRouted(ch, [ch], (3, 3), 1, ch, ch, (H, W), window,
                      max_depth=max_depth, min_depth=min_depth,
                      hard_routing=True, positional=True)

x = torch.randn(b, window, ch, H, W)
y = torch.randn(b, ch, H, W)

# 1. forward shape
model.train()
out = model(x)
assert out.shape == (b, ch, H, W), f"bad output shape {out.shape}"

# 2. input-independence: same depth distribution for every batch item
dist = model.router.last_dist  # (b, window, K)
assert torch.allclose(dist[0], dist[1]) and torch.allclose(dist[0], dist[2]), \
    "positional router should be identical across the batch"

# 3. gradient reaches pos_logits
loss = nn.MSELoss()(out, y) + 0.01 * model.depth_budget()
loss.backward()
g = model.router.pos_logits.grad
assert g is not None and g.abs().sum() > 0, "pos_logits got no gradient"

# 4. derived quantities
ed = model.expected_depth()          # (b, window)
assert ed.shape == (b, window)
assert min_depth <= float(ed.mean()) <= max_depth
db = model.depth_budget()
assert 0 < float(db) <= 1
ent = model.router_entropy()
assert 0 <= float(ent) <= 1.0001

# 5. eval mode -> hard one-hot per position
model.eval()
with torch.no_grad():
    model(x)
p = model.last_depth_probs  # (b, window, K)
assert torch.allclose(p.sum(-1), torch.ones(b, window))
assert ((p == 0) | (p == 1)).all(), "eval routing should be one-hot"

print("ALL POSITIONAL CHECKS PASSED")
print("  init mean depth (untrained, ~uniform):", round(float(ed.mean()), 3))
print("  eval per-position depth:",
      (min_depth + p[0].argmax(-1)).tolist())
