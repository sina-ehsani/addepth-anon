"""PUBLIC cross-dataset (ICLR): PredRNN depth-routing on TaxiBJ (OpenSTL benchmark).

Standard TaxiBJ: 4 years (BJ13-16) of citywide taxi in/out-flow on a 32x32 grid, 2
channels, 30-min slots. Task (OpenSTL): predict the next 4 frames from the previous 4
(4->4, autoregressive). We route encode-depth over the 4 input positions.

Data: data/taxibj_grid/BJ{13,14,15,16}.h5 (keys 'data' (T,2,32,32) float, 'date'),
downloaded from the ASTIR mirror. Windows are built WITHIN each year (never across the
gaps between years), pooled in chronological year order; test = last `test_slots`, val =
the preceding `val_slots`, train = the rest (chronological, no leakage). Normalized to
[0,1] by the global max (relative deep-vs-router MSE is invariant to this).

Modes: deep (uniform full depth) | custom (--encode_depths) | routed (learned +optional
--init_schedule). Tuned recipe available via --sched onecycle --grad_clip. Mirrors
bench_predrnn.py so the model + train loops + convergence metrics are reused unchanged.
"""
import argparse, json, logging, sys, time, os, glob
import numpy as np, torch, torch.nn as nn, h5py
from torch.utils.data import Dataset, DataLoader, Subset
from src.models.PredRNN import VarDepthPredRNN, PredRNNRouted
from src.train_loop import train_model, train_model_routed, evaluate_mse


class TaxiBJ(Dataset):
    def __init__(self, years, index, win, hor):
        self.years, self.index, self.win, self.hor = years, index, win, hor
    def __len__(self): return len(self.index)
    def __getitem__(self, k):
        yi, s = self.index[k]; a = self.years[yi]
        X = a[s:s + self.win]                              # (win, 2, 32, 32)
        y = a[s + self.win:s + self.win + self.hor]        # (hor, 2, 32, 32)
        return X, y


def build_taxibj(data_dir, win, hor):
    years = []
    for y in (13, 14, 15, 16):
        f = os.path.join(data_dir, f"BJ{y}.h5")
        with h5py.File(f, "r") as h:
            years.append(np.array(h["data"], dtype=np.float32))   # (T,2,32,32)
    mx = max(a.max() for a in years)
    years = [a / mx for a in years]                                # [0,1]
    # windows WITHIN each year (no cross-year sequences), chronological year order
    index = [(yi, s) for yi, a in enumerate(years)
             for s in range(len(a) - win - hor + 1)]
    return years, index, mx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["deep", "custom", "routed"], required=True)
    p.add_argument("--encode_depths", type=int, nargs="+", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--data_dir", default="../data/taxibj_grid")
    p.add_argument("--window", type=int, default=4)     # input frames
    p.add_argument("--horizon", type=int, default=4)    # predicted frames
    p.add_argument("--test_slots", type=int, default=1344)  # ~last 4 weeks (4*7*48)
    p.add_argument("--val_slots", type=int, default=1344)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)     # max_depth
    p.add_argument("--min_depth", type=int, default=1)
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--sched", choices=["none", "onecycle"], default="none")
    p.add_argument("--grad_clip", type=float, default=None)
    # routed
    p.add_argument("--lambda_budget", type=float, default=0.0)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--entropy_weight", type=float, default=0.0)
    p.add_argument("--router_warmup_epochs", type=int, default=0)
    p.add_argument("--init_schedule", type=int, nargs="+", default=None)
    p.add_argument("--init_margin", type=float, default=3.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
    label = a.label or a.mode
    out = a.out or f"results/bench_taxibj_{label}_L{a.layers}_s{a.seed}.json"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [taxibj-{label}|h{a.hidden}L{a.layers}] %(message)s")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    years, index, mx = build_taxibj(a.data_dir, a.window, a.horizon)
    n = len(index)
    te_n, va_n = a.test_slots, a.val_slots
    tr_idx = list(range(0, n - te_n - va_n)); va_idx = list(range(n - te_n - va_n, n - te_n))
    te_idx = list(range(n - te_n, n))
    ds = TaxiBJ(years, index, a.window, a.horizon)
    dl = dict(num_workers=a.num_workers, pin_memory=(dev.type == "cuda"),
              persistent_workers=(a.num_workers > 0), prefetch_factor=(4 if a.num_workers else None))
    tr = DataLoader(Subset(ds, tr_idx), batch_size=a.batch_size, shuffle=True, **dl)
    va = DataLoader(Subset(ds, va_idx), batch_size=a.batch_size, shuffle=False, **dl)
    te = DataLoader(Subset(ds, te_idx), batch_size=a.batch_size, shuffle=False, **dl)
    xb, yb = next(iter(tr))
    logging.info(f"data max={mx:.0f} windows={n} | X {tuple(xb.shape)} y {tuple(yb.shape)} "
                 f"| tr/va/te {len(tr_idx)}/{len(va_idx)}/{len(te_idx)}")

    ckpt = f"checkpoints/taxibj_{label}_L{a.layers}_s{a.seed}.pt"
    depth_profile = None
    if a.mode == "routed":
        model = PredRNNRouted(2, a.hidden, a.layers, a.horizon, window=a.window, min_depth=a.min_depth,
                              ksize=a.ksize, tensor_shape=(32, 32), hard_routing=True, positional=True).to(dev)
        if a.init_schedule is not None:
            assert len(a.init_schedule) == a.window, f"--init_schedule needs {a.window} ints"
            with torch.no_grad():
                model.router.pos_logits.zero_()
                for i, d in enumerate(a.init_schedule):
                    model.router.pos_logits[i, d - a.min_depth] = a.init_margin
            logging.info(f"router INIT at {a.init_schedule} (margin={a.init_margin})")
        nparams = sum(x.numel() for x in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr))
                 if a.sched == "onecycle" else None)
        logging.info(f"mode=routed min={a.min_depth} max={a.layers} window={a.window} "
                     f"lambda={a.lambda_budget} sched={a.sched} params={nparams/1e6:.2f}M")
        t0 = time.time()
        model, _, val_losses, best_val, depth_profile, last_epoch = train_model_routed(
            model, nn.MSELoss(), opt, tr, va, te, a.epochs, a.val_interval, a.patience,
            logging, dev, lambda_budget=a.lambda_budget, tau_start=a.tau_start, tau_end=a.tau_end,
            entropy_weight=a.entropy_weight, router_warmup_epochs=a.router_warmup_epochs,
            scheduler=sched, sched_per_batch=(a.sched == "onecycle"), grad_clip=a.grad_clip,
            print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)
        depths = [round(x) for x in depth_profile]
        comp = (sum(depth_profile) + a.layers * a.horizon) / (a.layers * (a.window + a.horizon))
    else:
        model = VarDepthPredRNN(2, a.hidden, a.layers, a.horizon, ksize=a.ksize, tensor_shape=(32, 32)).to(dev)
        if a.mode == "custom":
            assert a.encode_depths and len(a.encode_depths) == a.window, f"--encode_depths needs {a.window}"
            model.set_encode_depths(a.encode_depths)
        depths = model.encode_depths or [a.layers] * a.window
        comp = model.encode_compute_frac(a.window)
        nparams = sum(x.numel() for x in model.parameters())
        logging.info(f"mode={a.mode} encode_depths={depths} compute%={100*comp:.1f} params={nparams/1e6:.2f}M")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr))
                 if a.sched == "onecycle" else None)
        t0 = time.time()
        model, _, val_losses, best_val, last_epoch = train_model(
            model, nn.MSELoss(), opt, tr, va, te, a.epochs, a.val_interval, a.patience,
            logging, dev, scheduler=sched, sched_per_batch=(a.sched == "onecycle"),
            grad_clip=a.grad_clip, print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)

    mse = evaluate_mse(model, te, dev)
    minutes = (time.time() - t0) / 60
    b2b = min(val_losses, key=lambda x: x[1])[0] if val_losses else None
    res = {"dataset": "taxibj", "mode": a.mode, "encode_depths": depths, "compute_frac": comp,
           "test_mse": mse, "best_val": best_val, "window": a.window, "horizon": a.horizon,
           "params_M": nparams / 1e6, "minutes": minutes, "batches_to_best": b2b,
           "epochs_trained": last_epoch, "min_per_epoch": minutes / max(1, last_epoch)}
    if depth_profile is not None:
        res["depth_profile"] = [round(x, 3) for x in depth_profile]
        res["mean_depth"] = sum(depth_profile) / len(depth_profile)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== TaxiBJ {label}: test MSE {mse:.6f} | compute {100*comp:.0f}% | "
          f"{nparams/1e6:.2f}M | {minutes:.1f} min ===")


if __name__ == "__main__":
    main()
