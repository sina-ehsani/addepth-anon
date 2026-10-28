#!/usr/bin/env python
"""Single source of truth for every number in the TMLR paper.

Reads results/*.json and prints the aggregated table numbers (per-pixel MSE
mean +/- std over seeds, inference compute%, learned depth profile, convergence).
Every table/figure in the paper MUST match this output -- no hardcoded numbers.

Usage: python scripts/collect_paper_numbers.py
"""
import json, glob, os, math

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def load(path):
    with open(path) as f:
        return json.load(f)


def mse(d):
    # MMNIST/patchify store per-pixel; taxi/precip store test_mse
    return d.get("test_mse_perpixel", d.get("test_mse"))


def agg(paths):
    """mean/std of per-pixel MSE + compute over a seed set; first file's profile."""
    ds = [load(p) for p in paths if os.path.exists(p)]
    if not ds:
        return None
    ms = [mse(x) for x in ds]
    comp = [x.get("compute_frac") for x in ds]
    prof = ds[0].get("depth_profile") or ds[0].get("encode_depths")
    b2b = [x.get("batches_to_best") for x in ds if x.get("batches_to_best") is not None]
    n = len(ms)
    mean = sum(ms) / n
    std = math.sqrt(sum((m - mean) ** 2 for m in ms) / (n - 1)) if n > 1 else 0.0  # sample std
    return dict(n=n, mean=mean, std=std, per_seed=ms,
                compute=sum(comp) / len(comp) if comp[0] is not None else None,
                profile=prof, batches_to_best=b2b,
                params_M=ds[0].get("params_M"))


def R(*names):
    return [os.path.join(RES, n) for n in names]


def seeds(prefix, suffix, ss=(0, 1, 2)):
    return [os.path.join(RES, f"{prefix}_s{s}{suffix}") for s in ss]


# label -> list of json paths (seed sets aggregated)
GROUPS = {
    "== PredRNN 5.1M pilot (MMNIST 10->10) ==": None,
    "  deep (3-seed)":        seeds("bench_predrnn_deep_L4", ".json"),
    "  deep_long (1-seed)":   R("bench_predrnn_deep_long_L4_s0.json"),
    "  ramp hand-set":        seeds("bench_predrnn_ramp_L4", ".json"),
    "  recency hand-set":     seeds("bench_predrnn_recency_L4", ".json"),
    "  router warm-start":    R("bench_predrnn_routed_initrec_lam0_L4_s0.json",
                               "bench_predrnn_routed_initrec_lam0_L4_s1.json"),

    "== PredRNN 20M scale (MMNIST, tuned recipe) ==": None,
    "  deep-tuned (3-seed)":  seeds("bench_predrnn_deepcyc_h128_L4", ".json"),
    "  router (3-seed)":      seeds("bench_predrnn_initreccyc_h128_L4", ".json"),
    "  ramp hand-set (3-seed)": seeds("bench_predrnn_rampcyc_h128_L4", ".json"),
    "  from-uniform (1)":     R("bench_predrnn_fromunicyc_h128_L4_s0.json"),
    "  [robustness] deep naive": R("bench_predrnn_deep_h128_L4_s0.json",
                                   "bench_predrnn_deeplong_h128_L4_s0.json"),
    "  [robustness] router naive": seeds("bench_predrnn_initrec_h128_L4", ".json"),

    "== PredRNN patchify (STRONG backbone; the boundary) ==": None,
    "  uniform-4 (3-seed)":   seeds("bench_predrnn_patch4_deep_L4", ".json"),
    "  uniform-6":            R("bench_predrnn_patch4_deep6_L6_s0.json"),
    "  uniform-8":            R("bench_predrnn_patch4_deep8_L8_s0.json"),
    "  router L4 warmstart (3-seed)": seeds("bench_predrnn_patch4_router_L4", ".json"),
    "  v1 prune8 l1e3":       R("bench_predrnn_patch4_prune8_l1e3_L8_s0.json"),
    "  v1 prune8 l2e3":       R("bench_predrnn_patch4_prune8_l2e3_L8_s0.json"),
    "  v2 ctrl firm (m10)":   R("bench_predrnn_patch4d_ctrl_firm_L8_s0.json"),
    "  v2 prune-from-8":      R("bench_predrnn_patch4d_prune8v_m3t05_l2e3_L8_s0.json"),
    "  v2 from-6":            R("bench_predrnn_patch4d_from6_m2t05_l1e3_L8_s0.json"),
    "  r8 grow-from-4":       R("bench_predrnn_patch4_r8grow_L8_s0.json"),

    "== Cross-dataset PredRNN ==": None,
    "  precip deep (3-seed)":   seeds("bench_precip_deep_7015_L6", ".json"),
    "  precip router (3-seed)": seeds("bench_precip_initrec_7015_L6", ".json"),
    "  precip grow+brake":      R("bench_precip_growbudget_lam2e-4_L6_s0.json"),
    "  taxi(pooled) deep (3-seed)":    seeds("bench_taxi_deep_pooled_L6", ".json"),
    "  taxi(pooled) router (3-seed)":  seeds("bench_taxi_initshallow_pooled_L6", ".json"),
    "  taxi(pooled) fromuniform": R("bench_taxi_fromuniform_pooled_L6_s0.json"),
    "  taxibj deep":            R("bench_taxibj_deep_L4_s0.json"),
    "  taxibj ramp":            R("bench_taxibj_ramp_L4_s0.json"),
    "  taxibj grow+brake":      R("bench_taxibj_growbudget_lam2e-4_L4_s0.json"),

    "== Uniform-depth controls (true L=3/L=2 models; compute_frac in JSON is vs own L) ==": None,
    "  5.1M uniform-3 (75%)":      seeds("bench_predrnn_uni3_L3", ".json"),
    "  20M uniform-3 (75%)":       seeds("bench_predrnn_uni3cyc_h128_L3", ".json"),
    "  precip uniform-3 (50%)":    seeds("bench_precip_uni3_7015_L3", ".json"),
    "  precip uniform-2 (33%)":    seeds("bench_precip_uni2_7015_L2", ".json"),
    "  taxi(pooled) uniform-3 (50%)": seeds("bench_taxi_uni3_pooled_L3", ".json"),
    "  taxi(pooled) uniform-2 (33%)": seeds("bench_taxi_uni2_pooled_L2", ".json"),
    "  taxibj uniform-3 (75%)":    seeds("bench_taxibj_uni3_L3", ".json"),
    "  patchify uniform-6 (3-seed)": seeds("bench_predrnn_patch4_deep6_L6", ".json"),
    "  patchify v1 prune8 l1e3 (3-seed)": seeds("bench_predrnn_patch4_prune8_l1e3_L8", ".json"),

    "== Grow-from-shallow lambda sweep (MMNIST 5.1M) ==": None,
    "  grow lam2e-4":         R("bench_predrnn_growbudget_lam2e-4_L4_s0.json"),
    "  grow lam5e-4":         R("bench_predrnn_growbudget_lam5e-4_L4_s0.json"),
    "  grow lam1e-3":         R("bench_predrnn_growbudget_lam1e-3_L4_s0.json"),
}


def fmt_prof(p):
    if not p:
        return "-"
    return "[" + ",".join(str(int(round(x))) for x in p) + "]"


def main():
    print("=" * 78)
    print("PAPER NUMBERS -- authoritative, from results/*.json")
    print("=" * 78)
    for label, paths in GROUPS.items():
        if paths is None:                      # section header
            print(f"\n{label}")
            continue
        a = agg(paths)
        if a is None:
            print(f"  {label:34s}  [MISSING JSON]")
            continue
        pm = f"{a['mean']:.5f}"
        if a["n"] > 1:
            pm += f" +/- {a['std']:.5f}"
        comp = f"{100*a['compute']:.0f}%" if a["compute"] is not None else "  -"
        steps = ""
        if a["profile"]:
            # total layer-steps for MMNIST-style 10->10 (encode + full/routed decode when present)
            steps = f" steps~{int(round(sum(a['profile'])))}"
        b2b = f" b2best={min(a['batches_to_best'])}" if a["batches_to_best"] else ""
        print(f"  {label:34s} n={a['n']} MSE={pm:22s} comp={comp:4s} "
              f"{fmt_prof(a['profile'])}{steps}{b2b}")
    print("\n(per-seed values available in the JSONs; std is sample std, n-1.)")


if __name__ == "__main__":
    main()
