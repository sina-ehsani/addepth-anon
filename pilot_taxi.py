"""Pilot: full-length single-seed taxi runs on the REAL preprocessed tensors.

Runs (sequentially, each with early stopping):
    convlstm  - ConvLSTM baseline
    bidepth   - BidirectionalDeepShallow (fixed linear depth schedule)
    rl        - AdDepth + Q-learning (loss-based state; the paper's method, bug-fixed)
    random    - ablation: random actions + rollback (no Q-learning)
    routed    - AdDepthRouted (differentiable router, AdDepth v2)

Also reports two zero-parameter persistence baselines (previous day, same day
last week) computed directly on the test split.

All models share the corrected chronological pipeline: holdout is split into
val (first half, drives early stopping / RL / router selection) and test
(second half, plain MSE, touched once).

Usage:
    python pilot_taxi.py                       # all models
    python pilot_taxi.py --models routed rl    # subset
"""

import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from src.dataset_processing import create_dataloader_taxi
from src.train_loop import (train_model, train_model_routed, train_model_with_RL,
                            evaluate_mse)
from src.models.ConvLSTM import ConvLSTM
from src.models.BiDepth import BidirectionalDeepShallow
from src.models.AdaptiveDepth import AdDepth, QLearningAgent
from src.models.DepthRouter import AdDepthRouted

H5_DEFAULT = "../all_tensors_7_zone2.h5"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def persistence_baselines(test_loader, device):
    """MSE of predicting the previous day / same day last week.

    Assumes pu_ts index -1 is the most recent historical day and index 0 the
    oldest (7 days before target) — verify against preprocessing if surprising.
    """
    se_prev = se_week = n = 0.0
    for pu_ts, _do, labels in test_loader:
        se_prev += ((pu_ts[:, -1].float() - labels.float()) ** 2).sum().item()
        se_week += ((pu_ts[:, 0].float() - labels.float()) ** 2).sum().item()
        n += labels.numel()
    return se_prev / n, se_week / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=H5_DEFAULT)
    parser.add_argument("--models", nargs="+",
                        default=["convlstm", "bidepth", "rl", "random", "routed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda_budget", type=float, default=0.01)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0,
                        help="epochs of uniform routing before the router engages")
    parser.add_argument("--out", default="results/pilot_results.json")
    args = parser.parse_args()

    logging.basicConfig(filename="pilot_taxi.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"== device: {device} | seed: {args.seed} ==")

    batch_size = 128
    train_loader, val_loader, test_loader = create_dataloader_taxi(
        batch_size, h5_path=args.h5, val_fraction=0.5)
    pu_ts, do, labels = next(iter(train_loader))
    print(f"train X {tuple(pu_ts.shape)}, standalone {tuple(do.shape)}, y {tuple(labels.shape)} | "
          f"val/test sizes: {len(val_loader.dataset)}/{len(test_loader.dataset)}")

    # Paper-scale config
    window = pu_ts.shape[1]            # 7
    channels = pu_ts.shape[2]          # 16
    tensor_shape = pu_ts.shape[-2:]    # (58, 14)
    hidden_dim = [channels]
    kernel_size = (3, 3)
    min_depth, max_depth, initial_depth = 1, 10, 5
    validation_interval = 5
    criterion = nn.MSELoss()

    results = {}

    mse_prev, mse_week = persistence_baselines(test_loader, device)
    results["persistence_prev_day"] = {"test_mse": mse_prev}
    results["persistence_last_week"] = {"test_mse": mse_week}
    print(f"\n[persistence] prev-day test MSE {mse_prev:.4f} | last-week test MSE {mse_week:.4f}\n")

    def common(model):
        return dict(criterion=criterion,
                    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
                    train_loader=train_loader, val_loader=val_loader,
                    test_loader=test_loader, num_epochs=args.epochs,
                    validation_interval=validation_interval, patience=args.patience,
                    logging=logging, device=device, print_losses=False)

    for name in args.models:
        torch.manual_seed(args.seed)
        t0 = time.time()
        ckpt = f"checkpoints/checkpoint_pilot_{name}.pt"
        extra = {}

        if name == "convlstm":
            model = ConvLSTM(channels, hidden_dim, kernel_size, len(hidden_dim),
                             channels, channels, tensor_shape, window).to(device)
            c = common(model)
            model, _, vl, best_val, _ = train_model(
                model, c["criterion"], c["optimizer"], train_loader, val_loader,
                test_loader, args.epochs, validation_interval, args.patience,
                logging, device, print_losses=False, checkpoint_path=ckpt)

        elif name == "bidepth":
            model = BidirectionalDeepShallow(channels, hidden_dim, kernel_size,
                                             len(hidden_dim), channels, channels,
                                             tensor_shape, window,
                                             initial_layers=max_depth).to(device)
            c = common(model)
            model, _, vl, best_val, _ = train_model(
                model, c["criterion"], c["optimizer"], train_loader, val_loader,
                test_loader, args.epochs, validation_interval, args.patience,
                logging, device, print_losses=False, checkpoint_path=ckpt)

        elif name in ("shallow", "deep"):
            # Fixed uniform-depth controls: bracket the routed model.
            # If 'shallow' (depth 1 everywhere) matches routed-all-1, depth is
            # simply unnecessary on this dataset and the router's contribution
            # is having discovered that.
            d = 1 if name == "shallow" else max_depth
            model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                            channels, channels, tensor_shape, window,
                            initial_layers=d).to(device)
            c = common(model)
            model, _, vl, best_val, _ = train_model(
                model, c["criterion"], c["optimizer"], train_loader, val_loader,
                test_loader, args.epochs, validation_interval, args.patience,
                logging, device, print_losses=False, checkpoint_path=ckpt)
            extra["fixed_depth"] = d

        elif name in ("rl", "random"):
            model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                            channels, channels, tensor_shape, window,
                            initial_layers=max_depth).to(device)
            if name == "rl":
                agent = QLearningAgent(window, init_loss=4.0)
            else:
                # Ablation: actions stay uniformly random forever; only the
                # rollback mechanism remains.
                agent = QLearningAgent(window, init_loss=4.0,
                                       exploration_rate=1.0, exploration_decay=1.0)
            c = common(model)
            model, _, vl, best_val, best_layers, _ = train_model_with_RL(
                model, c["criterion"], c["optimizer"], train_loader, test_loader,
                val_loader, args.epochs, validation_interval, args.patience,
                q_learning_agent=agent, window_size=window, max_depth=max_depth,
                initial_depth=initial_depth, logging=logging, device=device,
                min_depth=min_depth, Rl_start_step=10, loss_spike_threshold=1.1,
                print_losses=False, checkpoint_path=ckpt)
            extra["best_layers"] = best_layers

        elif name == "routed":
            model = AdDepthRouted(channels, hidden_dim, kernel_size, len(hidden_dim),
                                  channels, channels, tensor_shape, window,
                                  max_depth=max_depth, min_depth=min_depth).to(device)
            c = common(model)
            model, _, vl, best_val, depth_profile, _ = train_model_routed(
                model, c["criterion"], c["optimizer"], train_loader, val_loader,
                test_loader, args.epochs, validation_interval, args.patience,
                logging, device, lambda_budget=args.lambda_budget,
                tau_start=2.0, tau_end=0.5, entropy_weight=args.entropy_weight,
                router_warmup_epochs=args.warmup,
                print_losses=False, checkpoint_path=ckpt)
            extra["depth_profile"] = [round(d, 2) for d in depth_profile]

        elif name == "positional":
            # Learnable per-position depth schedule, NO input conditioning
            # (the recommended adaptive variant — see HANDOFF2.md). Same method as
            # on weather: hard STE routing, tau_end=0.1, entropy=0, warmup=0, all
            # intrinsic to positional. Cross-dataset claim: same method should
            # discover shallow-everywhere on taxi (depth doesn't matter here),
            # vs the recency ramp on precipitation — no manual tuning.
            model = AdDepthRouted(channels, hidden_dim, kernel_size, len(hidden_dim),
                                  channels, channels, tensor_shape, window,
                                  max_depth=max_depth, min_depth=min_depth,
                                  hard_routing=True, positional=True).to(device)
            c = common(model)
            model, _, vl, best_val, depth_profile, _ = train_model_routed(
                model, c["criterion"], c["optimizer"], train_loader, val_loader,
                test_loader, args.epochs, validation_interval, args.patience,
                logging, device, lambda_budget=args.lambda_budget,
                tau_start=2.0, tau_end=0.1, entropy_weight=0.0,
                router_warmup_epochs=0,
                print_losses=False, checkpoint_path=ckpt)
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

    print(f"\n=== Summary (test MSE, lower is better) ===")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["test_mse"]):
        print(f"  {name:22s} {r['test_mse']:.4f}")
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
