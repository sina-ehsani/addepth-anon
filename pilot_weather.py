"""Pilot: US precipitation runs on the GPU server (honest chronological pipeline).

Same harness as pilot_taxi.py, pointed at the NowCasting-derived tensor
(e.g. train_test_2015_2020.h5, key 'Precipitation', shape (T, H, W) at
half-hourly resolution). Task: 24h of history as 6 stacks of 8 half-hour
frames -> predict the next 4h (8 frames), matching the paper setup.

Models: convlstm, bidepth, rl, random, routed, positional, fixed_schedule,
        shallow, deep
Plus two zero-parameter references computed on the test split:
  - persistence: repeat the most recent 4h stack
  - zeros: predict no precipitation everywhere (sparse-target sanity check)

DIRECTION (see HANDOFF2.md): 'positional' is the recommended adaptive variant —
a learnable per-position depth schedule (no input conditioning, hard STE). The
input-conditioned 'routed' is kept as a negative-result ablation: the static
'fixed_schedule' control beat it on both accuracy and compute. Sweep
--lambda_budget on 'positional' for the accuracy/compute Pareto curve; target is
deep-6 accuracy (~0.461) at <100% compute. Recommended: --tau_end 0.1.

NOTE: --warmup defaults to 5 but results-4 showed warmup is HARMFUL (early
stopping fires on the untrained-router checkpoint). Pass --warmup 0 for 'routed'.
'positional' forces warmup=0 internally.

Examples:
    python pilot_weather.py --models positional --lambda_budget 0.001 --tau_end 0.1
    python pilot_weather.py --models positional fixed_schedule deep --crop 64 64
    for s in 0 1 2; do python pilot_weather.py --models positional --seed $s \
        --tau_end 0.1 --out pilot_w_pos_s$s.json; done
"""

import argparse
import json
import logging
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from src.dataset_processing import create_dataloaders
from src.train_loop import (train_model, train_model_routed, train_model_with_RL,
                            evaluate_mse)
from src.models.ConvLSTM import ConvLSTM
from src.models.BiDepth import BidirectionalDeepShallow
from src.models.AdaptiveDepth import AdDepth, QLearningAgent
from src.models.DepthRouter import AdDepthRouted


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_baselines(test_loader):
    """Persistence (repeat most recent input stack) and all-zeros MSE."""
    se_pers = se_zero = n = 0.0
    for X, y in test_loader:
        X, y = X.float(), y.float()
        se_pers += ((X[:, -1] - y) ** 2).sum().item()
        se_zero += (y ** 2).sum().item()
        n += y.numel()
    return se_pers / n, se_zero / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="../data/weather/train_test_2015_2020.h5")
    parser.add_argument("--key", default="Precipitation")
    parser.add_argument("--models", nargs="+",
                        default=["convlstm", "bidepth", "rl", "random",
                                 "routed", "shallow", "deep"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_budget", type=float, default=0.001)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tau_start", type=float, default=2.0,
                        help="initial Gumbel-softmax temperature (routed)")
    parser.add_argument("--tau_end", type=float, default=0.5,
                        help="final Gumbel-softmax temperature (routed); lower = sharper routing")
    parser.add_argument("--crop", type=int, nargs=2, metavar=("H", "W"),
                        help="center-crop the spatial grid (reduced pilot, e.g. 64 64)")
    parser.add_argument("--max_frames", type=int,
                        help="use only the first N time frames (reduced pilot)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--val_interval", type=int,
                        help="override validation interval (#train batches between "
                             "validations); default is int(10000/batch_size)")
    parser.add_argument("--amp", action="store_true",
                        help="train with bf16 autocast on CUDA (halves activation memory; "
                             "reported test MSE stays fp32 for comparability)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes (data loading was the "
                             "bottleneck: 0->8 workers gives ~3x throughput)")
    parser.add_argument("--tag", default="",
                        help="suffix for checkpoint filenames; set a unique value to run "
                             "concurrent same-model variants without checkpoint collisions")
    parser.add_argument("--router_hard", action="store_true",
                        help="routed: straight-through HARD Gumbel routing in training so the "
                             "train forward uses the real discrete depth (matches hard eval); "
                             "fixes the soft-mixture train/eval mismatch")
    parser.add_argument("--fixed_schedule", type=int, nargs="+", default=[2, 2, 3, 3, 5, 6],
                        help="fixed_schedule model: frozen per-position depth list "
                             "(len must equal window); a static control vs the learned router")
    parser.add_argument("--window", type=int, default=6,
                        help="number of history stacks (each = 8 half-hour frames = 4h). "
                             "Default 6 = 24h. Larger spans more history to test whether the "
                             "learned positional schedule shows periodic (diurnal) depth.")
    parser.add_argument("--out", default="results/pilot_weather_results.json")
    args = parser.parse_args()

    # Single unified log stream per run: route logging + prints (+ tqdm on stderr)
    # to stdout so a shell redirect captures everything in order, per run. The
    # run id is baked into every line so concurrent/merged logs stay parseable.
    run_id = f"{'+'.join(args.models)}|s{args.seed}{args.tag}"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [{run_id}] %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    if device.type == "cuda":
        device = torch.device(f"cuda:{args.gpu}")

    logging.info(f"===== RUN {run_id} =====")
    logging.info(f"device={device} models={args.models} seed={args.seed} "
                 f"epochs={args.epochs} patience={args.patience} bs={args.batch_size} "
                 f"lr={args.lr} amp={args.amp} num_workers={args.num_workers}")
    logging.info(f"routed cfg: warmup={args.warmup} lambda_budget={args.lambda_budget} "
                 f"entropy_weight={args.entropy_weight} tau={args.tau_start}->{args.tau_end} "
                 f"router_hard={args.router_hard}")

    with h5py.File(args.h5, "r") as f:
        if args.key not in f:
            raise KeyError(f"'{args.key}' not in {args.h5}; available: {list(f.keys())}")
        # fp32 storage. (Tested fp16 to cut DataLoader transfer bandwidth, but at
        # num_workers=8 the run is GPU-compute-bound — workers already hide loading
        # — so fp16 gave no speedup and only perturbed results. The .float() casts
        # in the train loops are no-ops here but keep things robust.)
        data = np.array(f[args.key], dtype=np.float32)
    if args.max_frames:
        data = data[: args.max_frames]
    if args.crop:
        H, W = data.shape[-2:]
        ch, cw = args.crop
        r0, c0 = (H - ch) // 2, (W - cw) // 2
        data = data[:, r0: r0 + ch, c0: c0 + cw]
    print(f"data: {data.shape} (T, H, W), max={data.max():.2f}, mean={data.mean():.4f}")

    # Paper setup: 6 stacks x 8 half-hour frames = 24h history, predict next 8.
    # --window extends history (12 = 48h, 18 = 72h) to probe periodic depth.
    frame_number, window = 8, args.window
    train_loader, val_loader, test_loader = create_dataloaders(
        data, frame_number=frame_number, window_size=window,
        batch_size=args.batch_size, train_split=0.85, val_split=0.05,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    xb, yb = next(iter(train_loader))
    print(f"X {tuple(xb.shape)}, y {tuple(yb.shape)} | "
          f"train/val/test samples: {len(train_loader.dataset)}/"
          f"{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    channels = frame_number
    tensor_shape = data.shape[-2:]
    hidden_dim = [channels]
    kernel_size = (3, 3)
    min_depth, max_depth, initial_depth = 1, 6, 3
    validation_interval = args.val_interval or max(1, int(10000 / args.batch_size))
    criterion = nn.MSELoss()

    results = {}
    mse_pers, mse_zero = reference_baselines(test_loader)
    results["persistence_last_stack"] = {"test_mse": mse_pers}
    results["zeros"] = {"test_mse": mse_zero}
    print(f"\n[reference] persistence {mse_pers:.4f} | zeros {mse_zero:.4f}\n")

    for name in args.models:
        torch.manual_seed(args.seed)
        t0 = time.time()
        ckpt = f"checkpoints/checkpoint_pilotw_{name}{args.tag}.pt"
        extra = {}

        if name == "convlstm":
            model = ConvLSTM(channels, hidden_dim, kernel_size, len(hidden_dim),
                             channels, channels, tensor_shape, window).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model, _, vl, best_val, _ = train_model(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)

        elif name == "bidepth":
            model = BidirectionalDeepShallow(channels, hidden_dim, kernel_size,
                                             len(hidden_dim), channels, channels,
                                             tensor_shape, window,
                                             initial_layers=max_depth).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model, _, vl, best_val, _ = train_model(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)

        elif name in ("shallow", "deep"):
            d = 1 if name == "shallow" else max_depth
            model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                            channels, channels, tensor_shape, window,
                            initial_layers=d).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model, _, vl, best_val, _ = train_model(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)
            extra["fixed_depth"] = d

        elif name == "fixed_schedule":
            # Static control: a frozen non-uniform per-position depth schedule
            # (no router). If the learned router can't beat this, input-
            # conditioning adds nothing over a hand-set recency schedule.
            sched = args.fixed_schedule
            if len(sched) != window:
                raise ValueError(f"--fixed_schedule needs {window} ints, got {sched}")
            model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                            channels, channels, tensor_shape, window,
                            initial_layers=max_depth).to(device)  # allocate all conv layers
            model.update_dynamic_layer_counts(list(sched))        # freeze the schedule
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model, _, vl, best_val, _ = train_model(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)
            extra["fixed_schedule"] = list(sched)

        elif name in ("rl", "random"):
            model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                            channels, channels, tensor_shape, window,
                            initial_layers=max_depth).to(device)
            if name == "rl":
                agent = QLearningAgent(window, init_loss=4.0)
            else:
                agent = QLearningAgent(window, init_loss=4.0,
                                       exploration_rate=1.0, exploration_decay=1.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model, _, vl, best_val, best_layers, _ = train_model_with_RL(
                model, criterion, optimizer, train_loader, test_loader, val_loader,
                args.epochs, validation_interval, args.patience,
                q_learning_agent=agent, window_size=window, max_depth=max_depth,
                initial_depth=initial_depth, logging=logging, device=device,
                min_depth=min_depth, Rl_start_step=10, loss_spike_threshold=1.1,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)
            extra["best_layers"] = best_layers

        elif name == "routed":
            model = AdDepthRouted(channels, hidden_dim, kernel_size, len(hidden_dim),
                                  channels, channels, tensor_shape, window,
                                  max_depth=max_depth, min_depth=min_depth,
                                  hard_routing=args.router_hard).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model, _, vl, best_val, depth_profile, _ = train_model_routed(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                lambda_budget=args.lambda_budget, tau_start=args.tau_start,
                tau_end=args.tau_end,
                entropy_weight=args.entropy_weight,
                router_warmup_epochs=args.warmup,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)
            extra["depth_profile"] = [round(d, 2) for d in depth_profile]
            extra["router_hard"] = args.router_hard
            extra["mean_depth"] = round(sum(depth_profile) / len(depth_profile), 3)

        elif name == "positional":
            # Learnable per-position depth schedule, NO input conditioning
            # (the recommended adaptive variant — see HANDOFF2.md). Keeps the
            # differentiable schedule-search (Gumbel + STE + budget penalty) but
            # drops frame content, since the optimal depth here is positional,
            # not per-sample. Hard STE routing always on (train forward == hard
            # eval). Warmup is forced to 0: it was shown to make early stopping
            # fire on the untrained-router checkpoint (results-4), and a
            # positional schedule has no cross-input rich-get-richer risk.
            model = AdDepthRouted(channels, hidden_dim, kernel_size, len(hidden_dim),
                                  channels, channels, tensor_shape, window,
                                  max_depth=max_depth, min_depth=min_depth,
                                  hard_routing=True, positional=True).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            # tau_end and entropy_weight are intrinsic to positional, not knobs
            # (like warmup=0): tau_end=0.1 sharpens the STE late in training, and
            # entropy_weight=0 because a single shared schedule has no cross-input
            # collapse risk — the anti-collapse bonus would only fight the schedule
            # committing to a depth per position. lambda_budget stays the swept dial.
            model, _, vl, best_val, depth_profile, _ = train_model_routed(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                lambda_budget=args.lambda_budget, tau_start=args.tau_start,
                tau_end=0.1,
                entropy_weight=0.0,
                router_warmup_epochs=0,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)
            extra["depth_profile"] = [round(d, 2) for d in depth_profile]
            extra["mean_depth"] = round(sum(depth_profile) / len(depth_profile), 3)
            extra["positional"] = True

        else:
            print(f"unknown model '{name}', skipping")
            continue

        test_mse = evaluate_mse(model, test_loader, device)
        n_params = sum(p.numel() for p in model.parameters())
        results[name] = {"test_mse": test_mse, "best_val": best_val,
                         "params": n_params, "minutes": (time.time() - t0) / 60,
                         **extra}
        print(f"[{name:9s}] test MSE {test_mse:.4f} | best val {best_val:.4f} | "
              f"{n_params:,} params | {(time.time()-t0)/60:.1f} min | {extra or ''}")

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Summary (test MSE, lower is better) ===")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["test_mse"]):
        print(f"  {name:24s} {r['test_mse']:.4f}")
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
