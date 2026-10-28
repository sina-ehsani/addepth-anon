"""Paper figures for the learned-depth-router (ICLR) work.

Fig 1  learned per-position depth distribution on MMNIST (the interpretable ramp):
       softmax(pos_logits) heatmap, positions x depths, argmax overlaid.
Fig 2  cross-dataset discovered schedules (3 panels): MMNIST ramp / precip depth-
       on-recent / taxi shallow -- "different schedule per dataset, no tuning".
Fig 3  accuracy-vs-compute thesis in one axis: relative test MSE (router/deep) vs
       inference compute%, one point per dataset (router sits at low compute,
       accuracy <= or ~= deep).

CPU only (matplotlib + torch for the checkpoint). Writes PDF (LaTeX) + PNG (preview).
Palette: Okabe-Ito (colorblind-safe), fixed order. Sequential heatmap = single hue.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = HERE
os.makedirs(OUT, exist_ok=True)

# --- Okabe-Ito (CVD-safe), fixed roles ---
ROUTER = "#0072B2"   # blue  -- the method (hero)
DEEP   = "#D55E00"   # vermillion -- full-depth baseline
HANDSET= "#009E73"   # green -- hand-set reduced
INK    = "#222222"; MUTED = "#888888"; GRID = "#DDDDDD"

mpl.rcParams.update({
    "font.size": 9, "font.family": "sans-serif", "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# --- authoritative number access: pull straight from results/*.json --------------
RES = os.path.join(ROOT, "results")

def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)

def mse_of(name):
    d = _load(name)
    return d.get("test_mse_perpixel", d.get("test_mse"))

def sched_of(name):
    d = _load(name)
    return d.get("depth_profile") or d.get("encode_depths")

def mean_mse(names):
    v = [mse_of(n) for n in names]
    return sum(v) / len(v)


# ============ Fig 1: MMNIST learned depth distribution (pos_logits heatmap) ======
def fig1_ramp():
    ckpt = os.path.join(ROOT, "checkpoints/predrnn_initreccyc_h128_L4_s0.pt")  # 20M scale router
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    pos = sd["router.pos_logits"]                 # (window=10, K=4)
    P = F.softmax(pos, dim=-1).numpy()            # (10, 4) prob over depths 1..4
    W, K = P.shape
    argmax = P.argmax(1) + 1                       # selected depth per position

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    im = ax.imshow(P.T, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
    # argmax overlay (the discrete schedule the model deploys)
    ax.plot(range(W), argmax - 1, "-o", color=DEEP, lw=1.8, ms=5, mfc="white",
            mec=DEEP, mew=1.4, label="selected depth (argmax)")
    ax.set_xlabel("input position  (oldest → most recent)")
    ax.set_ylabel("processing depth")
    ax.set_xticks(range(W)); ax.set_yticks(range(K)); ax.set_yticklabels(range(1, K + 1))
    ax.set_title("Learned per-position depth (Moving MNIST, 20M)", fontsize=9.5, pad=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("router probability", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    save(fig, "fig1_mmnist_ramp")


# ============ Fig 2: cross-dataset discovered schedules (3 panels) ================
def fig2_cross_dataset():
    # schedules loaded from the authoritative router JSONs (NOT hardcoded); verdicts are
    # factual and honest -- MMNIST beats, precip/taxi MATCH deep (no "no-tuning" claim).
    # Second verdict line: vs the uniform-depth controls nearest the router's compute (drawn as
    # dotted lines; label gives each one's own inference compute).
    panels = [
        ("Moving MNIST (20M)", "bench_predrnn_initreccyc_h128_L4_s0.json", 4, "77% compute", "beats deep",
         [(3, "uniform-3 (75%)")], "ties uniform-3, more stable"),
        ("Precipitation",      "bench_precip_initrec_7015_L6_s0.json",      6, "40% compute", "matches deep",
         [(3, "uniform-3 (50%)"), (2, "uniform-2 (33%)")], "beats uniform-2 and -3"),
        ("NYC Taxi",           "bench_taxi_initshallow_pooled_L6_s0.json",  6, "33% compute", "matches deep",
         [(3, "uniform-3 (50%)"), (2, "uniform-2 (33%)")], "ties uniform-2 at equal compute"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for ax, (name, jf, maxd, comp, verdict, unis, uverdict) in zip(axes, panels):
        sched = [int(round(x)) for x in sched_of(jf)]
        x = np.arange(len(sched))
        ax.step(x, sched, where="mid", color=ROUTER, lw=2.0)
        ax.fill_between(x, 0, sched, step="mid", color=ROUTER, alpha=0.15)
        ax.plot(x, sched, "o", color=ROUTER, ms=5, mfc="white", mec=ROUTER, mew=1.4)
        ax.axhline(maxd, color=DEEP, lw=1.2, ls="--", alpha=0.8)
        ax.text(len(sched)-1, maxd+0.12, "full depth", color=DEEP, fontsize=7,
                ha="right", va="bottom")
        for d, lab in unis:
            ax.axhline(d, color=MUTED, lw=1.0, ls=":", zorder=1)
            ax.text(-0.4, d + 0.08, lab, color=MUTED, fontsize=6.5, ha="left", va="bottom")
        ax.set_title(name, fontsize=9.5, pad=30)
        ax.set_ylim(0, maxd + 0.8); ax.set_xlim(-0.5, len(sched)-0.5)
        ax.set_xlabel("position (old→new)", fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=7.5)
        ax.text(0.5, 1.02, f"{comp} · {verdict}\n{uverdict}", transform=ax.transAxes, fontsize=7.5,
                va="bottom", ha="center", color=MUTED, linespacing=1.3)
    axes[0].set_ylabel("learned depth")
    fig.suptitle("Learned depth schedules are recency-weighted and dataset-dependent",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, "fig2_cross_dataset_schedules")


# ==== Fig 3+4 COMBINED (one figure, two panels -> guaranteed identical scale/fonts) ====
def fig34_frontier_robust():
    # Rendered as ONE figure with two subplots so both panels share the exact same physical
    # scale (bbox_inches='tight' crops each standalone PDF differently, which is what made the
    # separate includes look mismatched). Numbers pulled from JSON.
    import statistics as st
    def ms(names):
        v = [mse_of(n) for n in names]
        return (sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
    S = lambda pre, ss=(0, 1, 2): [f"{pre}_s{s}.json" for s in ss]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # ---- Left: accuracy vs inference compute (relative to deep) ----
    specs = [
        ("Moving MNIST (20M)", 77, S("bench_predrnn_initreccyc_h128_L4"), S("bench_predrnn_deepcyc_h128_L4")),
        ("Precipitation", 40, S("bench_precip_initrec_7015_L6"), S("bench_precip_deep_7015_L6")),
        ("NYC Taxi", 33, S("bench_taxi_initshallow_pooled_L6"), S("bench_taxi_deep_pooled_L6")),
    ]
    axL.axhline(1.0, color=DEEP, lw=1.3, ls="--")
    axL.text(102, 1.0, "deep\nbaseline", color=DEEP, fontsize=7.5, va="center", ha="left")
    for name, comp, r, d in specs:
        (rm, rs, _), (dm, ds, _) = ms(r), ms(d)
        rel = rm / dm
        sr = rel * np.sqrt((rs / rm) ** 2 + (ds / dm) ** 2)  # ratio std via propagation
        axL.errorbar(comp, rel, yerr=sr, fmt="o", color=ROUTER, ms=9, mfc=ROUTER,
                     mec="white", mew=1.2, ecolor=ROUTER, elinewidth=1.4, capsize=4, zorder=3)
        dx, ha = {"NYC Taxi": (-9, "right")}.get(name, (9, "left"))
        axL.annotate(name, (comp, rel), textcoords="offset points",
                     xytext=(dx, 0), ha=ha, va="center", fontsize=8, color=INK)
    axL.set_xlabel("inference compute  (% of full depth)")
    axL.set_ylabel("test MSE  relative to deep")
    axL.set_xlim(14, 116); axL.set_ylim(0.80, 1.09)
    axL.set_title("Moderate backbones:\nrouter matches / beats deep at low compute", fontsize=9.0, pad=6)
    axL.text(0.5, 0.96, "↓ below line = beats deep   (bars: ±1 SD)",
             transform=axL.transAxes, fontsize=7, color=MUTED, ha="center", va="top")

    # ---- Right: recipe robustness (naive vs tuned, 20M) ----
    series = [
        ("deep",   DEEP,   (0.03528, 0.00213), (0.01640, 0.00110)),  # naive = mean(0.03315,0.03741)
        ("router", ROUTER, (0.02026, 0.00150), (0.01434, 0.00008)),
    ]
    x = [0, 1]
    for name, col, (n_m, n_s), (t_m, t_s) in series:
        axR.errorbar(x, [n_m, t_m], yerr=[n_s, t_s], color=col, lw=2.2, marker="o", ms=8,
                     mfc="white", mec=col, mew=1.6, capsize=4, ecolor=col, elinewidth=1.3, zorder=3)
        axR.annotate(name, (1, t_m), textcoords="offset points", xytext=(8, 0), va="center",
                     fontsize=9, color=col, fontweight="bold")
        drop = 100 * (n_m - t_m) / n_m
        axR.annotate(f"−{drop:.0f}%", (0.5, (n_m + t_m) / 2), textcoords="offset points",
                     xytext=(0, 6 if name == "deep" else -14), ha="center", fontsize=7.5, color=col)
    axR.set_xticks(x); axR.set_xticklabels(["naïve recipe\n(lr1e-3, 50ep)", "tuned recipe\n(lr5e-4, 1cyc)"])
    axR.set_ylabel("test MSE"); axR.set_xlim(-0.25, 1.4); axR.set_ylim(0.010, 0.040)
    axR.set_title("Recipe sensitivity:\ndeep collapses, router robust (20M)", fontsize=9.0, pad=6)
    axR.text(0.02, 0.02, "naïve router already beats naïve deep",
             transform=axR.transAxes, fontsize=7, color=MUTED, va="bottom")

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig34_frontier_robust")


# ============ Fig 5: THE BOUNDARY -- strong patchify backbone (the negative) ======
def fig5_boundary():
    # Accuracy vs ABSOLUTE layer-steps (encode + decode) on the strong patchify PredRNN.
    # abs_steps is a transparent deterministic function of the profile + decode convention
    # (v1: decode = full max_depth x 10; v2 route_decode: profile already covers both);
    # MSE is pulled from the JSON (authoritative). Uniform models form the Pareto frontier;
    # every compute-SAVING router point sits ABOVE-RIGHT of it (dominated).
    # (name, layer-steps, mean, sample std or 0 for single-seed points)
    import statistics as st
    def ms(pre, ss=(0, 1, 2)):
        v = [mse_of(f"{pre}_s{s}.json") for s in ss]
        return sum(v) / len(v), (st.stdev(v) if len(v) > 1 else 0.0)
    U = [("uniform-4", 80, *ms("bench_predrnn_patch4_deep_L4")),
         ("uniform-6", 120, *ms("bench_predrnn_patch4_deep6_L6")),
         ("uniform-8", 160, mse_of("bench_predrnn_patch4_deep8_L8_s0.json"), 0.0)]
    # v1 prune 1e-3: seeds keep 140/136/136 layer-steps -> mean 137
    Rr = [("router-4 (warm)", 57, *ms("bench_predrnn_patch4_router_L4")),
          ("v1 prune λ2e-3", 125, mse_of("bench_predrnn_patch4_prune8_l2e3_L8_s0.json"), 0.0),
          ("v1 prune λ1e-3", 137, *ms("bench_predrnn_patch4_prune8_l1e3_L8")),
          ("v2 prune-from-8", 145, mse_of("bench_predrnn_patch4d_prune8v_m3t05_l2e3_L8_s0.json"), 0.0),
          ("v2 from-6", 120, mse_of("bench_predrnn_patch4d_from6_m2t05_l1e3_L8_s0.json"), 0.0)]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ux = [s for _, s, _, _ in U]; uy = [m for _, _, m, _ in U]
    # uniform Pareto frontier line + shaded dominated region above it
    ax.plot(ux, uy, "-o", color=DEEP, lw=2.0, ms=8, mfc="white", mec=DEEP, mew=1.8,
            zorder=4, label="uniform-depth models (Pareto frontier)")
    ax.errorbar(ux, uy, yerr=[sd for *_, sd in U], fmt="none", ecolor=DEEP, elinewidth=1.2,
                capsize=3, zorder=3)
    for name, s, m, _ in U:
        ax.annotate(name, (s, m), textcoords="offset points", xytext=(4, -12),
                    fontsize=7.5, color=DEEP)
    # per-point label offsets to avoid collisions
    off = {"router-4 (warm)": (6, -12, "left"), "v1 prune λ2e-3": (-6, 6, "right"),
           "v1 prune λ1e-3": (8, 4, "left"), "v2 prune-from-8": (4, 8, "left"),
           "v2 from-6": (-8, 8, "left")}
    for name, s, m, sd in Rr:
        ax.errorbar(s, m, yerr=sd or None, fmt="s", color=ROUTER, ms=8, mfc=ROUTER, mec="white",
                    mew=1.2, ecolor=ROUTER, elinewidth=1.2, capsize=3, zorder=5)
        dx, dy, ha = off.get(name, (6, 5, "left"))
        ax.annotate(name, (s, m), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=7.2, color=ROUTER)
    ax.plot([], [], "s", color=ROUTER, ms=8, label="adaptive-depth router")
    ax.set_xlabel("inference compute  (absolute layer-steps, encode + decode)")
    ax.set_ylabel("test MSE  (per pixel)")
    ax.set_title("Strong backbone: adaptive depth is dominated by uniform depth\n"
                 "(every compute-saving router point lies above the uniform frontier)",
                 fontsize=9.0, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.set_xlim(50, 172); ax.set_ylim(top=0.01548, bottom=0.01372)
    ax.text(0.98, 0.70, "bars: ±1 SD over 3 seeds\n(no bar = single seed)", transform=ax.transAxes,
            fontsize=7, color=MUTED, ha="right", va="top")
    save(fig, "fig5_boundary")


# ============ Fig 6: grow-from-shallow discovery from a neutral init =============
def fig6_growshallow():
    # Same all-shallow (depth-1 everywhere) init + compute brake -> DIFFERENT dataset-
    # appropriate schedules emerge (non-circular discovery). Schedules from JSON.
    panels = [
        ("Moving MNIST", "bench_predrnn_growbudget_lam5e-4_L4_s0.json", 4, r"$\lambda{=}5\!\times\!10^{-4}$"),
        ("Precipitation", "bench_precip_growbudget_lam2e-4_L6_s0.json", 6, r"$\lambda{=}2\!\times\!10^{-4}$"),
        ("TaxiBJ",       "bench_taxibj_growbudget_lam2e-4_L4_s0.json", 4, r"$\lambda{=}2\!\times\!10^{-4}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for ax, (name, jf, maxd, lam) in zip(axes, panels):
        sched = [int(round(x)) for x in sched_of(jf)]
        x = np.arange(len(sched))
        ax.plot(x, [1] * len(sched), color=MUTED, lw=1.4, ls=":", label="all-shallow init")
        ax.step(x, sched, where="mid", color=HANDSET, lw=2.0, label="grown schedule")
        ax.fill_between(x, 1, sched, step="mid", color=HANDSET, alpha=0.15)
        ax.plot(x, sched, "o", color=HANDSET, ms=5, mfc="white", mec=HANDSET, mew=1.4)
        ax.axhline(maxd, color=DEEP, lw=1.1, ls="--", alpha=0.7)
        ax.text(len(sched) - 1, maxd + 0.1, "full depth", color=DEEP, fontsize=7, ha="right", va="bottom")
        ax.set_title(f"{name}  ({lam})", fontsize=9, pad=6)
        ax.set_ylim(0, maxd + 0.8); ax.set_xlim(-0.5, len(sched) - 0.5)
        ax.set_xlabel("position (old→new)", fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("depth")
    axes[2].legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Grow-from-shallow: one neutral init $+$ brake $\\to$ different schedules per dataset",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig6_growshallow")


if __name__ == "__main__":
    fig1_ramp()
    fig2_cross_dataset()
    fig34_frontier_robust()
    fig5_boundary()
    fig6_growshallow()
    print("all figures ->", OUT)
