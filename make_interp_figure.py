"""Interpretability figure: learned positional depth schedules across datasets.

The centerpiece for Sec 4.2 — the positional method auto-discovers a different
depth allocation per dataset, with no manual tuning:
  - precipitation: depth only on the most-recent stack (recency); ties deep-6 @~31% compute
  - moving mnist:  depth ramps on the most-recent 1-2 stacks; BEATS deep-6 @~25% compute
  - taxi:          shallow / unstructured (depth doesn't matter); ~shallow optimum

Reads the per-seed positional depth_profile from the pilot JSONs and plots
per-position selected depth (mean + individual seeds) per dataset.
"""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"          # per-run JSONs
FIG_DIR = "figures"              # output figures
os.makedirs(FIG_DIR, exist_ok=True)

# (title, glob, json_key, max_depth, caption)
PANELS = [
    ("Precipitation (weather)", "pilot_w_pos_lam0.001_s*.json", "positional", 6,
     "ties deep-6"),
    ("Moving MNIST", "pilot_mm_s*.json", "positional", 6,
     "beats deep-6"),
    ("NYC Taxi", "pilot_taxi_pos_s?.json", "positional", 10,
     "depth irrelevant → shallow"),
]


def load_profiles(pattern, key):
    profs = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        try:
            profs.append(json.load(open(f))[key]["depth_profile"])
        except (KeyError, json.JSONDecodeError):
            pass
    return [np.asarray(p, dtype=float) for p in profs]


fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 3.8))
for ax, (title, pat, key, maxd, cap) in zip(axes, PANELS):
    profs = load_profiles(pat, key)
    if not profs:
        ax.set_title(f"{title}\n(no data)"); continue
    W = len(profs[0])
    x = np.arange(W)
    arr = np.vstack([p for p in profs if len(p) == W])
    mean = arr.mean(0)
    mean_depth = mean.mean()
    # individual seeds (faint), mean (bold)
    for p in arr:
        ax.plot(x, p, color="0.7", lw=1, alpha=0.7, marker="o", ms=3, zorder=1)
    ax.plot(x, mean, color="C0", lw=2.5, marker="o", ms=6, zorder=3,
            label=f"mean (depth {mean_depth:.1f})")
    ax.axhline(maxd, ls="--", color="C3", lw=1, zorder=0, label=f"max depth ({maxd})")
    ax.axhline(1, ls=":", color="0.5", lw=1, zorder=0)
    ax.set_ylim(0.5, maxd + 0.5)
    ax.set_xticks(x)
    ax.set_xlabel("window position\n(oldest → most recent)")
    ax.set_title(f"{title}\n{cap} · ≈{100*mean_depth/maxd:.0f}% compute", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("selected depth\n(# conv layers)")
fig.suptitle("Learned positional depth schedules: the router allocates depth per dataset, no tuning",
             fontsize=12, y=1.04)
fig.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(FIG_DIR, "interp_schedules.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "interp_schedules.pdf"), bbox_inches="tight")
print(f"saved {FIG_DIR}/interp_schedules.png / .pdf")
for title, pat, key, maxd, cap in PANELS:
    profs = load_profiles(pat, key)
    if profs:
        arr = np.vstack(profs)
        print(f"  {title}: n={len(profs)} mean_per_pos={[round(v,2) for v in arr.mean(0)]}")
