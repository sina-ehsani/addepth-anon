"""Pilot: Moving MNIST (standard public video-prediction benchmark).

Cross-dataset generalization check for the positional depth schedule. Same
models/machinery as pilot_weather.py, on the canonical Moving MNIST sequences
(mnist_test_seq.npy, (20, 10000, 64, 64) uint8). Default task: 9 stacks of 2
frames (18 frames history) -> predict the next 2 frames.

Models: convlstm, bidepth, shallow, deep, fixed_schedule, rl, random, routed,
positional (the method). Reference baselines: persistence (repeat last input
stack) and zeros.

Examples:
    python pilot_mmnist.py --models positional deep shallow convlstm
    python pilot_mmnist.py --models positional --seed 0 --out pilot_mm_pos_s0.json
"""
import argparse
import json
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from src.dataset_processing import create_dataloaders_mmnist
from src.train_loop import (train_model, train_model_routed, train_model_with_RL,
                            evaluate_mse)
from src.models.ConvLSTM import ConvLSTM
from src.models.BiDepth import BidirectionalDeepShallow
from src.models.AdaptiveDepth import AdDepth, QLearningAgent
from src.models.DepthRouter import AdDepthRouted


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reference_baselines(test_loader):
    """Persistence (repeat the most recent input stack) and all-zeros MSE."""
    se_pers = se_zero = n = 0.0
    for X, y in test_loader:
        X, y = X.float(), y.float()
        se_pers += ((X[:, -1] - y) ** 2).sum().item()
        se_zero += (y ** 2).sum().item()
        n += y.numel()
    return se_pers / n, se_zero / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", default="../mnist_test_seq.npy")
    parser.add_argument("--models", nargs="+",
                        default=["positional", "deep", "shallow", "convlstm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_budget", type=float, default=0.001)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--tau_start", type=float, default=2.0)
    parser.add_argument("--tau_end", type=float, default=0.5)
    parser.add_argument("--frame_number", type=int, default=2)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--router_hard", action="store_true")
    parser.add_argument("--fixed_schedule", type=int, nargs="+", default=None,
                        help="frozen per-position depth (len must == window); "
                             "default = recency ramp [1,...,1,6]")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--tag", default="")
    parser.add_argument("--max_seqs", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default="results/pilot_mmnist_results.json")
    args = parser.parse_args()

    run_id = f"mm:{'+'.join(args.models)}|s{args.seed}{args.tag}"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [{run_id}] %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    if device.type == "cuda":
        device = torch.device(f"cuda:{args.gpu}")

    frame_number, window = args.frame_number, args.window
    logging.info(f"===== RUN {run_id} =====")
    logging.info(f"device={device} models={args.models} seed={args.seed} "
                 f"frame_number={frame_number} window={window} epochs={args.epochs} "
                 f"amp={args.amp} num_workers={args.num_workers}")

    train_loader, val_loader, test_loader = create_dataloaders_mmnist(
        args.npy, frame_number=frame_number, window_size=window,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), max_seqs=args.max_seqs)
    xb, yb = next(iter(train_loader))
    print(f"X {tuple(xb.shape)}, y {tuple(yb.shape)} | train/val/test seqs: "
          f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    channels = frame_number
    tensor_shape = (xb.shape[-2], xb.shape[-1])
    hidden_dim = [channels]
    kernel_size = (3, 3)
    min_depth, max_depth, initial_depth = 1, 6, 3
    validation_interval = args.val_interval or max(1, int(2000 / args.batch_size))
    criterion = nn.MSELoss()

    results = {}
    mse_pers, mse_zero = reference_baselines(test_loader)
    results["persistence_last_stack"] = {"test_mse": mse_pers}
    results["zeros"] = {"test_mse": mse_zero}
    print(f"\n[reference] persistence {mse_pers:.4f} | zeros {mse_zero:.4f}\n")

    for name in args.models:
        torch.manual_seed(args.seed)
        t0 = time.time()
        ckpt = f"checkpoints/checkpoint_pilotmm_{name}{args.tag}.pt"
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
            sched = args.fixed_schedule or ([1] * (window - 1) + [max_depth])
            if len(sched) != window:
                raise ValueError(f"--fixed_schedule needs {window} ints, got {sched}")
            model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                            channels, channels, tensor_shape, window,
                            initial_layers=max_depth).to(device)
            model.update_dynamic_layer_counts(list(sched))
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
                agent = QLearningAgent(window, init_loss=0.1)
            else:
                agent = QLearningAgent(window, init_loss=0.1,
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

        elif name in ("routed", "positional"):
            positional = (name == "positional")
            model = AdDepthRouted(channels, hidden_dim, kernel_size, len(hidden_dim),
                                  channels, channels, tensor_shape, window,
                                  max_depth=max_depth, min_depth=min_depth,
                                  hard_routing=(args.router_hard or positional),
                                  positional=positional).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            # positional: tau_end=0.1, entropy=0, warmup=0 (intrinsic, as in weather)
            tau_end = 0.1 if positional else args.tau_end
            ent_w = 0.0 if positional else args.entropy_weight
            warm = 0 if positional else args.warmup
            model, _, vl, best_val, depth_profile, _ = train_model_routed(
                model, criterion, optimizer, train_loader, val_loader, test_loader,
                args.epochs, validation_interval, args.patience, logging, device,
                lambda_budget=args.lambda_budget, tau_start=args.tau_start,
                tau_end=tau_end, entropy_weight=ent_w, router_warmup_epochs=warm,
                print_losses=False, checkpoint_path=ckpt, use_amp=args.amp)
            extra["depth_profile"] = [round(d, 2) for d in depth_profile]
            extra["mean_depth"] = round(sum(depth_profile) / len(depth_profile), 3)
            extra["positional"] = positional

        else:
            print(f"unknown model '{name}', skipping")
            continue

        test_mse = evaluate_mse(model, test_loader, device)
        n_params = sum(p.numel() for p in model.parameters())
        results[name] = {"test_mse": test_mse, "best_val": best_val,
                         "params": n_params, "minutes": (time.time() - t0) / 60, **extra}
        print(f"[{name:14s}] test MSE {test_mse:.5f} | best val {best_val:.5f} | "
              f"{n_params:,} params | {(time.time()-t0)/60:.1f} min | {extra or ''}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Summary (test MSE, lower is better) ===")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["test_mse"]):
        print(f"  {name:24s} {r['test_mse']:.5f}")
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
