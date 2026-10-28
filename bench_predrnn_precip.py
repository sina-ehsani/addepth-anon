"""Cross-dataset (ICLR): PredRNN depth-routing on PRECIPITATION.

Same question as MMNIST but on the precip pipeline (train_test_2015_2020.h5,
key 'Precipitation', (T,512,256)). Setup mirrors the ConvLSTM pilot: window=6
history stacks of frame_number=8 half-hour frames (24h), predict the next 8
(single-shot). max_depth=6 (as the ConvLSTM AdDepth used). Depth is routed over
the 6 encode positions.

  --mode deep    : uniform full depth ([max_depth]*window)
  --mode custom  : hand-set --encode_depths (len==window)
  --mode routed  : learned positional router (+ optional --init_schedule prior)

512x256 is heavy for ST-LSTM, so default --crop 128 128 (center) keeps it
tractable; deep/custom/routed all use the SAME crop => internally valid
depth-schedule comparison. Not leaderboard-comparable (cropped, small model).
"""
import argparse, json, logging, sys, time, os
import numpy as np, torch, torch.nn as nn, h5py
from src.models.PredRNN import VarDepthPredRNN, PredRNNRouted
from src.train_loop import train_model, train_model_routed, evaluate_mse
from src.dataset_processing import create_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["deep", "custom", "routed"], required=True)
    p.add_argument("--encode_depths", type=int, nargs="+", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--h5", default="../train_test_2015_2020.h5")
    p.add_argument("--key", default="Precipitation")
    p.add_argument("--crop", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--train_split", type=float, default=0.70)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--stride", type=int, default=1,
                   help="subsample sample indices (full temporal span, fewer near-dup samples)")
    p.add_argument("--frame_number", type=int, default=8)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=6)        # max_depth
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
    # routed knobs
    p.add_argument("--lambda_budget", type=float, default=0.0)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--entropy_weight", type=float, default=0.0)
    p.add_argument("--router_warmup_epochs", type=int, default=0)
    p.add_argument("--init_schedule", type=int, nargs="+", default=None,
                   help="init router at this per-position prior (warm-start)")
    p.add_argument("--init_margin", type=float, default=3.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
    label = a.label or a.mode
    out = a.out or f"results/bench_precip_{label}_L{a.layers}_s{a.seed}.json"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [precip-{label}|h{a.hidden}L{a.layers}] %(message)s")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    with h5py.File(a.h5, "r") as f:
        d = f[a.key]
        fmax = a.max_frames or d.shape[0]
        if a.crop:  # crop during the read (full data is 55GB; the crop is ~7GB)
            H, W = d.shape[-2:]; ch, cw = a.crop
            r0, c0 = (H - ch) // 2, (W - cw) // 2
            data = np.array(d[:fmax, r0:r0 + ch, c0:c0 + cw], dtype=np.float32)
        else:
            data = np.array(d[:fmax], dtype=np.float32)
    tensor_shape = tuple(data.shape[-2:])
    logging.info(f"data {data.shape} (T,H,W) crop={a.crop} max={data.max():.2f} mean={data.mean():.4f}")

    tr, va, te = create_dataloaders(
        data, frame_number=a.frame_number, window_size=a.window, batch_size=a.batch_size,
        train_split=a.train_split, val_split=a.val_split, stride=a.stride,
        num_workers=a.num_workers, pin_memory=(dev.type == "cuda"))
    xb, yb = next(iter(tr))
    logging.info(f"X {tuple(xb.shape)} y {tuple(yb.shape)} | tr/va/te "
                 f"{len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)}")
    # in_ch = frame_number (stacks as channels); single-shot predict => t_out=1, collapse_out.

    ckpt = f"checkpoints/precip_{label}_L{a.layers}_s{a.seed}.pt"
    depth_profile = None
    if a.mode == "routed":
        model = PredRNNRouted(a.frame_number, a.hidden, a.layers, 1, window=a.window,
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
                     f"lambda={a.lambda_budget} tau={a.tau_start}->{a.tau_end} params={nparams/1e6:.2f}M")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        t0 = time.time()
        model, _, val_losses, best_val, depth_profile, last_epoch = train_model_routed(
            model, nn.MSELoss(), opt, tr, va, te, a.epochs, a.val_interval, a.patience,
            logging, dev, lambda_budget=a.lambda_budget, tau_start=a.tau_start, tau_end=a.tau_end,
            entropy_weight=a.entropy_weight, router_warmup_epochs=a.router_warmup_epochs,
            print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)
        depths = [round(x) for x in depth_profile]
        comp = (sum(depth_profile) + a.layers * 1) / (a.layers * (a.window + 1))
        logging.info(f"learned depth_profile={[round(x,2) for x in depth_profile]} "
                     f"mean={sum(depth_profile)/len(depth_profile):.2f} compute%={100*comp:.1f}")
    else:
        model = VarDepthPredRNN(a.frame_number, a.hidden, a.layers, 1, ksize=a.ksize,
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
            model, nn.MSELoss(), opt, tr, va, te, a.epochs, a.val_interval, a.patience,
            logging, dev, print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)

    mse = evaluate_mse(model, te, dev)
    minutes = (time.time() - t0) / 60
    # convergence metrics (hardware-independent training-benefit axis): the val_interval
    # (batch count) at which best val was reached, and epochs trained.
    batches_to_best = min(val_losses, key=lambda x: x[1])[0] if val_losses else None
    res = {"dataset": "precip", "mode": a.mode, "encode_depths": depths, "compute_frac": comp,
           "test_mse": mse, "best_val": best_val, "crop": a.crop, "window": a.window,
           "params_M": nparams / 1e6, "minutes": minutes,
           "batches_to_best": batches_to_best, "epochs_trained": last_epoch,
           "min_per_epoch": minutes / max(1, last_epoch)}
    if depth_profile is not None:
        res["depth_profile"] = [round(x, 3) for x in depth_profile]
        res["mean_depth"] = sum(depth_profile) / len(depth_profile)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== PRECIP {label}: test MSE {mse:.5f} | compute {100*comp:.0f}% | "
          f"{nparams/1e6:.2f}M | {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
