"""HANDOFF3 Phase-0 go/no-go: does reduced encode-depth preserve PredRNN accuracy?

Trains a (small) PredRNN on Moving MNIST 10->10 and compares:
  --mode uniform : full depth at every encode step  ([L]*T_in)
  --mode reduced : hand-set recency schedule        ([1,...,1,L])
SAME model/recipe; we want the RELATIVE accuracy gap vs the compute saved.
  GO   if reduced ~= uniform accuracy at compute% < 100  -> build the full thing
  NO-GO if reduced markedly worse -> per-position depth is a weak-backbone artifact
We do NOT chase PredRNN's published number here (that's Phase 1); only the gap matters.
Run both modes (e.g. on two GPUs), then compare results/bench_predrnn_*.json.
"""
import argparse, json, logging, sys, time, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from src.models.PredRNN import VarDepthPredRNN, PredRNNRouted
from src.train_loop import train_model, train_model_routed, evaluate_mse


def _patchify(x, p):
    # (T, C, H, W) -> (T, C*p*p, H//p, W//p)  (space-to-depth, OpenSTL PredRNN style)
    T, C, H, W = x.shape
    x = x.reshape(T, C, H // p, p, W // p, p).transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(T, C * p * p, H // p, W // p)


class MM10(Dataset):
    def __init__(self, data, t_in, t_out, patch_size=1):
        self.data, self.i, self.o, self.p = data, t_in, t_out, patch_size
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, k):
        seq = self.data[k]; H, W = seq.shape[1], seq.shape[2]
        X = seq[:self.i].reshape(self.i, 1, H, W)
        y = seq[self.i:self.i + self.o].reshape(self.o, 1, H, W)
        if self.p > 1:                       # patchify both to (T, p*p, H//p, W//p)
            X, y = _patchify(X, self.p), _patchify(y, self.p)
        return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["uniform", "reduced", "custom", "routed"], required=True)
    p.add_argument("--encode_depths", type=int, nargs="+", default=None,
                   help="mode=custom: explicit per-position encode depths (len must == t_in)")
    # --- routed (learned positional depth router) ---
    p.add_argument("--min_depth", type=int, default=1)
    p.add_argument("--window", type=int, default=None, help="router positions (default t_in)")
    p.add_argument("--lambda_budget", type=float, default=0.01)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--entropy_weight", type=float, default=0.0)
    p.add_argument("--router_warmup_epochs", type=int, default=0)
    p.add_argument("--hard_routing", action="store_true", default=True)
    p.add_argument("--soft_routing", dest="hard_routing", action="store_false",
                   help="fallback: soft-mix + anneal instead of hard STE")
    p.add_argument("--positional", action="store_true", default=True)
    p.add_argument("--input_conditioned", dest="positional", action="store_false",
                   help="ablation: input-conditioned router instead of positional")
    p.add_argument("--init_schedule", type=int, nargs="+", default=None,
                   help="init router pos_logits biased toward this per-position schedule "
                        "(diagnostic: does the router STAY here or drift to uniform?)")
    p.add_argument("--init_margin", type=float, default=3.0,
                   help="logit margin for --init_schedule (moderate => biased but can still move)")
    p.add_argument("--route_decode", action="store_true",
                   help="v2: also route per-decode-step depth (decode was full-depth in v1)")
    p.add_argument("--decode_init", type=int, nargs="+", default=None,
                   help="decode-position init schedule (len t_out); default = all max_depth when --route_decode")
    p.add_argument("--label", default=None, help="name for output/checkpoint (default: mode)")
    p.add_argument("--npy", default="../mnist_test_seq.npy")
    p.add_argument("--t_in", type=int, default=10)
    p.add_argument("--t_out", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)        # max_depth
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--sched", choices=["none", "onecycle"], default="none",
                   help="onecycle (OpenSTL-style) steps per batch over epochs*len(train)")
    p.add_argument("--grad_clip", type=float, default=None, help="max grad-norm (OpenSTL uses ~1)")
    p.add_argument("--patch_size", type=int, default=1, help="OpenSTL space-to-depth patchify (4 = strong PredRNN)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
    label = a.label or a.mode
    out = a.out or f"results/bench_predrnn_{label}_L{a.layers}_s{a.seed}.json"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [predrnn-{label}|h{a.hidden}L{a.layers}] %(message)s")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    arr = np.transpose(np.load(a.npy), (1, 0, 2, 3)).astype(np.float32) / 255.0
    ds = MM10(arr, a.t_in, a.t_out, patch_size=a.patch_size)
    in_ch = a.patch_size * a.patch_size          # space-to-depth channels (1 ch input)
    tshape = (64 // a.patch_size, 64 // a.patch_size)
    dl = dict(num_workers=a.num_workers, pin_memory=True,
              persistent_workers=(a.num_workers > 0), prefetch_factor=(4 if a.num_workers else None))
    tr = DataLoader(Subset(ds, range(0, 8000)), batch_size=a.batch_size, shuffle=True, **dl)
    va = DataLoader(Subset(ds, range(8000, 9000)), batch_size=a.batch_size, shuffle=False, **dl)
    te = DataLoader(Subset(ds, range(9000, 10000)), batch_size=a.batch_size, shuffle=False, **dl)

    ckpt = f"checkpoints/predrnn_{label}_L{a.layers}_s{a.seed}.pt"
    depth_profile = None
    if a.mode == "routed":
        window = a.window or a.t_in
        model = PredRNNRouted(in_ch, a.hidden, a.layers, a.t_out, window=window, min_depth=a.min_depth,
                              ksize=a.ksize, tensor_shape=tshape,
                              hard_routing=a.hard_routing, positional=a.positional,
                              route_decode=a.route_decode).to(dev)
        if a.init_schedule is not None:
            assert len(a.init_schedule) == window, f"--init_schedule needs {window} ints"
            # decode init: explicit --decode_init, else warm-start decode at full depth
            # (prune-from-deep: start at the strong full-depth decode, let budget trim it).
            dec_init = None
            if a.route_decode:
                dec_init = a.decode_init or [a.layers] * a.t_out
                assert len(dec_init) == a.t_out, f"--decode_init needs {a.t_out} ints"
            with torch.no_grad():
                model.router.pos_logits.zero_()
                for i, d in enumerate(a.init_schedule):
                    model.router.pos_logits[i, d - a.min_depth] = a.init_margin
                if dec_init is not None:
                    for t, d in enumerate(dec_init):
                        model.router.pos_logits[window + t, d - a.min_depth] = a.init_margin
            logging.info(f"router pos_logits INITIALIZED encode={a.init_schedule} "
                         f"decode={dec_init} (margin={a.init_margin})")
        nparams = sum(x.numel() for x in model.parameters())
        logging.info(f"mode=routed min_depth={a.min_depth} max_depth={a.layers} window={window} "
                     f"lambda={a.lambda_budget} tau={a.tau_start}->{a.tau_end} hard={a.hard_routing} "
                     f"positional={a.positional} params={nparams/1e6:.2f}M")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr))
                 if a.sched == "onecycle" else None)
        t0 = time.time()
        model, _, val_losses, best_val, depth_profile, last_epoch = train_model_routed(
            model, nn.MSELoss(), opt, tr, va, te, a.epochs, a.val_interval, a.patience,
            logging, dev, lambda_budget=a.lambda_budget, tau_start=a.tau_start, tau_end=a.tau_end,
            entropy_weight=a.entropy_weight, router_warmup_epochs=a.router_warmup_epochs,
            scheduler=sched, sched_per_batch=(a.sched == "onecycle"), grad_clip=a.grad_clip,
            print_losses=False, checkpoint_path=ckpt, use_amp=a.amp)
        # learned schedule + its inference compute% (eval = discrete argmax depths)
        depths = [round(x) for x in depth_profile]
        mean_depth = sum(depth_profile) / len(depth_profile)
        if a.route_decode:
            # depth_profile covers encode (t_in) + decode (t_out); both routed.
            comp = sum(depth_profile) / (a.layers * (a.t_in + a.t_out))
            enc_d, dec_d = depth_profile[:a.t_in], depth_profile[a.t_in:]
            logging.info(f"learned ENCODE={[round(x,2) for x in enc_d]} "
                         f"DECODE={[round(x,2) for x in dec_d]} "
                         f"mean_depth={mean_depth:.2f} inference-compute%={100*comp:.1f}")
        else:
            # encode routed, decode full-depth (v1)
            comp = (sum(depth_profile) + a.layers * a.t_out) / (a.layers * (a.t_in + a.t_out))
            logging.info(f"learned depth_profile={[round(x,2) for x in depth_profile]} "
                         f"mean_depth={mean_depth:.2f} inference-compute%={100*comp:.1f}")
    else:
        model = VarDepthPredRNN(in_ch, a.hidden, a.layers, a.t_out, ksize=a.ksize, tensor_shape=tshape).to(dev)
        if a.mode == "reduced":
            model.set_encode_depths([1] * (a.t_in - 1) + [a.layers])   # recency: deep only on last frame
        elif a.mode == "custom":
            assert a.encode_depths and len(a.encode_depths) == a.t_in, \
                f"--encode_depths needs {a.t_in} ints, got {a.encode_depths}"
            model.set_encode_depths(a.encode_depths)
        # uniform: leave encode_depths=None (all layers every step)
        depths = model.encode_depths or [a.layers] * a.t_in
        comp = model.encode_compute_frac(a.t_in)
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

    mse_px = evaluate_mse(model, te, dev)
    minutes = (time.time() - t0) / 60
    batches_to_best = min(val_losses, key=lambda x: x[1])[0] if val_losses else None
    res = {"mode": a.mode, "encode_depths": depths, "compute_frac": comp,
           "test_mse_perpixel": mse_px, "test_mse_openstl": mse_px * 64 * 64,
           "best_val": best_val, "params_M": nparams / 1e6, "minutes": minutes,
           "batches_to_best": batches_to_best, "epochs_trained": last_epoch,
           "min_per_epoch": minutes / max(1, last_epoch)}
    if depth_profile is not None:
        res["depth_profile"] = [round(x, 3) for x in depth_profile]
        res["mean_depth"] = sum(depth_profile) / len(depth_profile)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== PredRNN {a.mode}: per-pixel MSE {mse_px:.5f} | OpenSTL-conv {mse_px*4096:.1f} "
          f"| compute {100*comp:.0f}% | {nparams/1e6:.2f}M | {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
