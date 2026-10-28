"""Aggregate per-seed pilot JSONs into a mean +/- std table.

Reads files matching a glob (default the 3-seed top-cluster runs) and prints,
per model, the mean and sample std of test MSE across seeds. Seed-0 results
are pulled from the original full-pilot JSONs; seeds 1-2 from the *_s{seed}.json
runs. A model key is whatever appears inside each JSON (convlstm, deep, routed,
...); the routed warmup=0 variant is reported separately as 'routed_w0'.

Usage:
    python aggregate_seeds.py                 # default file set below
    python aggregate_seeds.py 'pilot_w_seed_*.json'
"""
import glob
import json
import sys
from collections import defaultdict


# (label, file, json_key) — seed is read from the JSON's own runs where present,
# else inferred from filename. We list files explicitly so seed-0 (from the
# original runs) and seeds 1-2 (new) are combined cleanly.
DEFAULT_FILES = [
    # seed 0 (reuse original full-pilot runs)
    ("bidepth",   "results/pilot_w_full_bidepth.json",          "bidepth"),
    ("deep",      "results/pilot_w_full_deep.json",             "deep"),
    ("rl",        "results/pilot_w_full_rl.json",               "rl"),
    ("routed_w0", "results/pilot_w_abl_routed_warm0.json",      "routed"),
]


def collect(files):
    """label -> list of test_mse across available seeds."""
    by_model = defaultdict(list)
    # explicit seed-0 set
    for label, f, key in files:
        try:
            d = json.load(open(f))
            by_model[label].append(d[key]["test_mse"])
        except (FileNotFoundError, KeyError):
            print(f"  [warn] missing {label} seed0 ({f})")
    # seeds 1-2 from the new per-seed runs: pilot_w_seed_{label}_s{n}.json
    for label, _, key in files:
        for path in sorted(glob.glob(f"results/pilot_w_seed_{label}_s*.json")):
            try:
                by_model[label].append(json.load(open(path))[key]["test_mse"])
            except (KeyError, json.JSONDecodeError):
                print(f"  [warn] unreadable {path}")
    return by_model


def main():
    files = DEFAULT_FILES
    by_model = collect(files)
    print(f"\n3-SEED TEST MSE (mean +/- std, n seeds)")
    print(f"  {'model':12s} {'mean':>8s} {'std':>8s}  n   values")
    rows = []
    for label, vals in by_model.items():
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        rows.append((mean, label, std, n, vals))
    for mean, label, std, n, vals in sorted(rows):
        vstr = ", ".join(f"{v:.4f}" for v in vals)
        print(f"  {label:12s} {mean:8.4f} {std:8.4f}  {n}   [{vstr}]")
    print()


if __name__ == "__main__":
    main()
