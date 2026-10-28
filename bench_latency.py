"""Inference latency of trained checkpoints (wall-clock complement to the layer-step proxy).

Architecture (in_ch, hidden, L, router window, decode routing) is inferred from each
checkpoint's state_dict; hand-set (custom) schedules come from the run's result JSON,
routed schedules from the router's argmax (the deployed eval fast path). Weights do not
change the timing, but loading the real checkpoints guarantees we time the deployed model.

Measures forward latency in eval/no_grad with CUDA events: warmup, then N timed iters,
reports median and p10/p90 in ms, at batch sizes 1 and 16. Run on an IDLE GPU only --
timings under contention are meaningless.

Usage: CUDA_VISIBLE_DEVICES=7 python bench_latency.py [--iters 200] [--bs 1 16] [--only mmnist20 ...]
"""
import argparse, json, os, statistics, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.models.PredRNN import VarDepthPredRNN, PredRNNRouted

# dataset -> (T_in, T_out, (H, W) after patchify, collapse_out)
GEOM = {
    "mmnist":  (10, 10, (64, 64), False),
    "patch4":  (10, 10, (16, 16), False),
    "precip":  (6, 1, (128, 128), True),
    "taxi":    (7, 1, (58, 14), True),
    "taxibj":  (4, 4, (32, 32), False),
}

# group -> list of (label, dataset, checkpoint, result json). Seed 0 of each config:
# latency depends on architecture + schedule, not on the seed.
C, R = "checkpoints/", "results/"
RUNS = {
    "mmnist20": [
        ("full depth (L4)",     "mmnist", C + "predrnn_deepcyc_h128_L4_s0.pt",    R + "bench_predrnn_deepcyc_h128_L4_s0.json"),
        ("ramp (hand-set)",     "mmnist", C + "predrnn_rampcyc_h128_L4_s0.pt",    R + "bench_predrnn_rampcyc_h128_L4_s0.json"),
        ("router (warm-start)", "mmnist", C + "predrnn_initreccyc_h128_L4_s0.pt", R + "bench_predrnn_initreccyc_h128_L4_s0.json"),
        ("uniform-3",           "mmnist", C + "predrnn_uni3cyc_h128_L3_s0.pt",    R + "bench_predrnn_uni3cyc_h128_L3_s0.json"),
    ],
    "patchify": [
        ("uniform-4",            "patch4", C + "predrnn_patch4_deep_L4_s0.pt",   R + "bench_predrnn_patch4_deep_L4_s0.json"),
        ("uniform-6",            "patch4", C + "predrnn_patch4_deep6_L6_s0.pt",  R + "bench_predrnn_patch4_deep6_L6_s0.json"),
        ("uniform-8",            "patch4", C + "predrnn_patch4_deep8_L8_s0.pt",  R + "bench_predrnn_patch4_deep8_L8_s0.json"),
        ("router-4 (warm-start)", "patch4", C + "predrnn_patch4_router_L4_s0.pt", R + "bench_predrnn_patch4_router_L4_s0.json"),
        ("v1 encode-prune 1e-3", "patch4", C + "predrnn_patch4_prune8_l1e3_L8_s0.pt", R + "bench_predrnn_patch4_prune8_l1e3_L8_s0.json"),
        ("v1 encode-prune 2e-3", "patch4", C + "predrnn_patch4_prune8_l2e3_L8_s0.pt", R + "bench_predrnn_patch4_prune8_l2e3_L8_s0.json"),
        ("v2 prune enc+dec",     "patch4", C + "predrnn_patch4d_prune8v_m3t05_l2e3_L8_s0.pt", R + "bench_predrnn_patch4d_prune8v_m3t05_l2e3_L8_s0.json"),
        ("v2 bidirectional from-6", "patch4", C + "predrnn_patch4d_from6_m2t05_l1e3_L8_s0.pt", R + "bench_predrnn_patch4d_from6_m2t05_l1e3_L8_s0.json"),
        ("v2 control (firm)",    "patch4", C + "predrnn_patch4d_ctrl_firm_L8_s0.pt", R + "bench_predrnn_patch4d_ctrl_firm_L8_s0.json"),
    ],
    "precip": [
        ("full depth (L6)", "precip", C + "precip_deep_7015_L6_s0.pt",    R + "bench_precip_deep_7015_L6_s0.json"),
        ("router",          "precip", C + "precip_initrec_7015_L6_s0.pt", R + "bench_precip_initrec_7015_L6_s0.json"),
        ("uniform-3",       "precip", C + "precip_uni3_7015_L3_s0.pt",    R + "bench_precip_uni3_7015_L3_s0.json"),
    ],
    "taxi": [
        ("full depth (L6)", "taxi", C + "taxi_deep_pooled_L6_s0.pt",        R + "bench_taxi_deep_pooled_L6_s0.json"),
        ("router",          "taxi", C + "taxi_initshallow_pooled_L6_s0.pt", R + "bench_taxi_initshallow_pooled_L6_s0.json"),
        ("uniform-3",       "taxi", C + "taxi_uni3_pooled_L3_s0.pt",        R + "bench_taxi_uni3_pooled_L3_s0.json"),
    ],
    "taxibj": [
        ("full depth (L4)", "taxibj", C + "taxibj_deep_L4_s0.pt", R + "bench_taxibj_deep_L4_s0.json"),
        ("ramp (hand-set)", "taxibj", C + "taxibj_ramp_L4_s0.pt", R + "bench_taxibj_ramp_L4_s0.json"),
        ("uniform-3",       "taxibj", C + "taxibj_uni3_L3_s0.pt", R + "bench_taxibj_uni3_L3_s0.json"),
    ],
}


def build(ds, ckpt, res_json, dev):
    t_in, t_out, shape, collapse = GEOM[ds]
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    L = len({k.split(".")[1] for k in sd if k.startswith("cells.")})
    w = sd["cells.0.conv_x.weight"]
    hidden, in_ch, ksize = w.shape[0] // 7, w.shape[1], w.shape[2]
    if "router.pos_logits" in sd:
        rows, K = sd["router.pos_logits"].shape
        min_depth = L - K + 1
        route_decode = rows == t_in + t_out
        m = PredRNNRouted(in_ch, hidden, L, t_out, window=t_in, min_depth=min_depth, ksize=ksize,
                          tensor_shape=shape, hard_routing=True, positional=True,
                          collapse_out=collapse, route_decode=route_decode)
        m.load_state_dict(sd)
        d = (min_depth + sd["router.pos_logits"].argmax(-1)).tolist()
        enc, dec = d[:t_in], (d[t_in:] if route_decode else [L] * t_out)
    else:
        m = VarDepthPredRNN(in_ch, hidden, L, t_out, ksize=ksize, tensor_shape=shape, collapse_out=collapse)
        m.load_state_dict(sd)
        enc = json.load(open(res_json))["encode_depths"] if os.path.exists(res_json) else [L] * t_in
        m.set_encode_depths(enc)
        dec = [L] * t_out
    return m.to(dev).eval(), (in_ch, shape, t_in), enc, dec, L


@torch.no_grad()
def time_model(m, in_ch, shape, t_in, bs, iters, warmup, dev):
    x = torch.randn(bs, t_in, in_ch, *shape, device=dev)
    for _ in range(warmup):
        m(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); m(x); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return statistics.median(ts), ts[len(ts) // 10], ts[(9 * len(ts)) // 10]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--bs", type=int, nargs="+", default=[1, 16])
    p.add_argument("--only", nargs="+", default=None, help=f"groups: {list(RUNS)}")
    p.add_argument("--out", default="results/latency.json")
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | fp32, eval, no_grad")
    out = []
    for grp, runs in RUNS.items():
        if a.only and grp not in a.only:
            continue
        print(f"\n== {grp} ==")
        base = {}
        for label, ds, ckpt, rj in runs:
            if not os.path.exists(ckpt):
                print(f"  {label:26s} [missing checkpoint {ckpt}]")
                continue
            m, (in_ch, shape, t_in), enc, dec, L = build(ds, ckpt, rj, dev)
            steps = sum(enc) + sum(dec)
            row = dict(group=grp, label=label, ckpt=ckpt, L=L, encode=enc, decode=dec, layer_steps=steps)
            for bs in a.bs:
                med, p10, p90 = time_model(m, in_ch, shape, t_in, bs, a.iters, a.warmup, dev)
                base.setdefault(bs, med)          # first row of each group is the reference
                row[f"bs{bs}_ms"] = dict(median=med, p10=p10, p90=p90, rel=med / base[bs])
            out.append(row)
            lat = "  ".join(f"bs{bs}: {row[f'bs{bs}_ms']['median']:7.2f} ms "
                            f"({100*row[f'bs{bs}_ms']['rel']:5.1f}%)" for bs in a.bs)
            print(f"  {label:26s} steps={steps:4d}  {lat}")
            del m; torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
