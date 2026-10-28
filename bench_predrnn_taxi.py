"""Cross-dataset (ICLR): PredRNN depth-routing on NYC TAXI.

Taxi is window=7 pickup timesteps -> predict the next pickup frame (predict-1),
each frame 16 channels x 58 x 14. (The dropoff stream `do` that the ConvLSTM
AdDepth fused is IGNORED in this v1 — we route depth over the 7 pickup steps and
predict pickup autoregressively; this keeps the SAME task for deep/custom/routed
so the depth-schedule comparison is internally valid. Fusing `do` is future work.)

Depth is routed over the 7 encode positions. The ConvLSTM finding on taxi was
"shallow schedules win" (opposite of precip), so the warm-start prior here is a
SHALLOW one — the cross-dataset test is whether the router refines to taxi's
shallow optimum, i.e. discovers a DIFFERENT schedule than precip with no tuning.

  --mode deep    : uniform full depth
  --mode custom  : hand-set --encode_depths (len==window=7)
  --mode routed  : learned positional router (+ optional --init_schedule prior)
"""
import argparse, json, logging, sys, time, os
import numpy as np, torch, torch.nn as nn
from src.models.PredRNN import VarDepthPredRNN, PredRNNRouted
from src.train_loop import train_model, train_model_routed, evaluate_mse
from src.dataset_processing import create_dataloader_taxi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["deep", "custom", "routed"], required=True)
    p.add_argument("--encode_depths", type=int, nargs="+", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--h5", default="../all_tensors_7_zone2.h5")
    p.add_argument("--pool_resplit", action="store_true",
                   help="pool train+holdout chronologically, 70/15/15 (bigger ~130 test vs 72)")
    p.add_argument("--window", type=int, default=7)        # pu_ts timesteps
    p.add_argument("--in_ch", type=int, default=16)        # channels per frame
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=6)        # max_depth
    p.add_argument("--min_depth", type=int, default=1)
    p.add_argument("--ksize", type=int, default=3)         # small spatial (58x14)
    p.add_argument("--lognorm", action="store_true", default=True,
                   help="log1p-normalize counts (default on; counts are heavy-tailed)")
    p.add_argument("--raw", dest="lognorm", action="store_false")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_interval", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--amp", action="store_true")
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
    out = a.out or f"results/bench_taxi_{label}_L{a.layers}_s{a.seed}.json"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [taxi-{label}|h{a.hidden}L{a.layers}] %(message)s")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    tr, va, te = create_dataloader_taxi(a.batch_size, h5_path=a.h5, val_fraction=0.5,
                                        pool_resplit=a.pool_resplit)
    # loader yields (pu_ts, do, pu): pu_ts (b,7,16,58,14), do (b,16,58,14), pu (b,16,58,14).
    # We route over pu_ts (window=7, in_ch=16), predict pu (single-shot t_out=1). `do` ignored.
    pu_ts, do, pu = next(iter(tr))
    tensor_shape = tuple(pu_ts.shape[-2:])
    logging.info(f"pu_ts {tuple(pu_ts.shape)} pu {tuple(pu.shape)} shape={tensor_shape} "
                 f"tr/va/te {len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)} lognorm={a.lognorm}")

    # Wrap loaders to (X, y) pairs (drop `do`), float + optional log1p. train loops
    # accept the 2-tuple path; we pre-map to avoid touching train_loop.
    def wrap(loader):
        for pu_ts, do, pu in loader:
            X = pu_ts.float(); y = pu.float()
            if a.lognorm:
                X = torch.log1p(X); y = torch.log1p(y)
            yield X, y
    class Wrapped:
        def __init__(self, loader): self.loader = loader
        def __iter__(self): return wrap(self.loader)
        def __len__(self): return len(self.loader)
        @property
        def dataset(self): return self.loader.dataset
    trw, vaw, tew = Wrapped(tr), Wrapped(va), Wrapped(te)

    ckpt = f"checkpoints/taxi_{label}_L{a.layers}_s{a.seed}.pt"
    depth_profile = None
    if a.mode == "routed":
        model = PredRNNRouted(a.in_ch, a.hidden, a.layers, 1, window=a.window,
                              min_depth=a.min_depth, ksize=a.ksize, tensor_shape=tensor_shape,
                              hard_routing=True, positional=True, collapse_out=True).to(dev)
        if a.init_schedule is not None:
            assert len(a.init_schedule) == a.window, f"--init_schedule needs {a.window} ints"
            with torch.no_grad():
                model.router.pos_logits.zero_()
                for i, d in enumerate(a.init_schedule):
                    model.router.pos_logits[i, d - a.min_depth] = a.init_margin
            logging.info(f"router INIT at {a.init_schedule} (margin={a.init_margin})")
        nparams = sum(x.numel() for x in model.parameters())
        logging.info(f"mode=routed min={a.min_depth} max={a.layers} window={a.window} "
                     f"lambda={a.lambda_budget} params={nparams/1e6:.2f}M")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        t0 = time.time()
        model, _, val_losses, best_val, depth_profile, last_epoch = train_model_routed(
            model, nn.MSELoss(), opt, trw, vaw, tew, a.epochs, a.val_interval, a.patience,
            logging, dev, lambda_budget=a.lambda_budget, tau_start=a.tau_start, tau_end=a.tau_end,
            entropy_weight=a.entropy_weight, router_warmup_epochs=a.router_warmup_epochs,
            print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)
        depths = [round(x) for x in depth_profile]
        comp = (sum(depth_profile) + a.layers * 1) / (a.layers * (a.window + 1))
        logging.info(f"learned depth_profile={[round(x,2) for x in depth_profile]} "
                     f"mean={sum(depth_profile)/len(depth_profile):.2f} compute%={100*comp:.1f}")
    else:
        model = VarDepthPredRNN(a.in_ch, a.hidden, a.layers, 1, ksize=a.ksize,
                                tensor_shape=tensor_shape, collapse_out=True).to(dev)
        if a.mode == "custom":
            assert a.encode_depths and len(a.encode_depths) == a.window, \
                f"--encode_depths needs {a.window} ints"
            model.set_encode_depths(a.encode_depths)
        depths = model.encode_depths or [a.layers] * a.window
        comp = model.encode_compute_frac(a.window)
        nparams = sum(x.numel() for x in model.parameters())
        logging.info(f"mode={a.mode} encode_depths={depths} compute%={100*comp:.1f} params={nparams/1e6:.2f}M")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        t0 = time.time()
        model, _, val_losses, best_val, last_epoch = train_model(
            model, nn.MSELoss(), opt, trw, vaw, tew, a.epochs, a.val_interval, a.patience,
            logging, dev, print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)

    mse = evaluate_mse(model, tew, dev)
    minutes = (time.time() - t0) / 60
    batches_to_best = min(val_losses, key=lambda x: x[1])[0] if val_losses else None
    res = {"dataset": "taxi", "mode": a.mode, "encode_depths": depths, "compute_frac": comp,
           "test_mse": mse, "best_val": best_val, "window": a.window, "lognorm": a.lognorm,
           "params_M": nparams / 1e6, "minutes": minutes,
           "batches_to_best": batches_to_best, "epochs_trained": last_epoch,
           "min_per_epoch": minutes / max(1, last_epoch)}
    if depth_profile is not None:
        res["depth_profile"] = [round(x, 3) for x in depth_profile]
        res["mean_depth"] = sum(depth_profile) / len(depth_profile)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== TAXI {label}: test MSE {mse:.5f} | compute {100*comp:.0f}% | "
          f"{nparams/1e6:.2f}M | {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
