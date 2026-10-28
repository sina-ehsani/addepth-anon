"""Accuracy-vs-compute curve for the routed (hard) lambda_budget sweep.

Reads pilot_w_curve_lam{lam}_s{seed}.json files, groups by lambda_budget, and
plots test MSE vs mean routed depth (the compute proxy: TimeDistributedRouted
applies `depth` shared conv layers per window position, so per-position FLOPs
scale with depth; the ConvLSTM backbone is a fixed cost). Mean depth is taken
from each run's `routed.depth_profile` (mean over the 6 positions == overall
mean expected depth). deep-6 and shallow-1 are drawn as horizontal references.

Usage: python plot_compute_curve.py            # default glob below
       python plot_compute_curve.py 'pilot_w_curve_lam*_s*.json'
"""
import glob
import json
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed-depth reference accuracies (3-seed-confirmed seed-0 numbers, same test split).
DEEP6_MSE = 0.4613
SHALLOW1_MSE = 0.4843
MAX_DEPTH = 6

LAM_RE = re.compile(r"lam([0-9.]+)_s(\d+)\.json$")


def _stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std, n


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_w_curve_lam*_s*.json"
    by_lam_mse = defaultdict(list)
    by_lam_depth = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        m = LAM_RE.search(path)
        if not m:
            print(f"  [warn] cannot parse lambda/seed from {path}")
            continue
        lam = float(m.group(1))
        try:
            r = json.load(open(path))["routed"]
        except (KeyError, json.JSONDecodeError):
            print(f"  [warn] no routed result in {path}")
            continue
        dp = r["depth_profile"]
        by_lam_mse[lam].append(r["test_mse"])
        by_lam_depth[lam].append(sum(dp) / len(dp))

    if not by_lam_mse:
        print(f"no matching runs for glob '{pattern}'")
        return

    lams = sorted(by_lam_mse)
    print(f"\nACCURACY vs COMPUTE (routed hard, sweep over lambda_budget)")
    print(f"  {'lambda':>8s} {'mse_mean':>9s} {'mse_std':>8s} {'depth':>7s} {'depth_std':>9s} {'compute%':>9s} n")
    xs, xerr, ys, yerr = [], [], [], []
    for lam in lams:
        mm, ms, n = _stats(by_lam_mse[lam])
        dm, ds, _ = _stats(by_lam_depth[lam])
        xs.append(dm); xerr.append(ds); ys.append(mm); yerr.append(ms)
        print(f"  {lam:8.4g} {mm:9.4f} {ms:8.4f} {dm:7.3f} {ds:9.3f} {100*dm/MAX_DEPTH:8.1f}% {n}")

    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o-", capsize=3, label="routed (hard)")
    for lam, x, y in zip(lams, xs, ys):
        plt.annotate(f"λ={lam:g}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.axhline(DEEP6_MSE, ls="--", color="green", label=f"deep-6 ({DEEP6_MSE})")
    plt.axhline(SHALLOW1_MSE, ls="--", color="red", label=f"shallow-1 ({SHALLOW1_MSE})")
    plt.xlabel("mean routed depth  (compute proxy; max=6)")
    plt.ylabel("test MSE")
    plt.title("Routed accuracy vs. average compute (precipitation, 3 seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/compute_curve.png", dpi=150)
    print("\nsaved figures/compute_curve.png")


if __name__ == "__main__":
    main()
