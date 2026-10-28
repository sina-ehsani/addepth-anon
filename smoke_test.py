"""Phase 1 smoke test: AdDepthRouted end-to-end on real taxi data (small scale).

Checks, in order:
  1. Data pipeline: chronological loaders from the hourly demand tensor.
  2. AdDepthRouted: forward/backward, shapes, budget loss, temperature annealing.
  3. train_model_routed: loss decreases, depth profile differentiates, hard-routing
     eval + plain-MSE test work.
  4. (--with-rl) the bug-fixed Q-learning loop still runs.
  5. ConvLSTM baseline for a rough quality reference at equal scale.

This is a functional test at toy scale — numbers are NOT paper numbers.

Usage:
    python prepare_taxi_smoke.py                  # once, to build smoke_data/
    python smoke_test.py [--epochs 8] [--with-rl]
"""

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from src.dataset_processing import create_dataloaders
from src.train_loop import train_model, train_model_routed, train_model_with_RL
from src.models.ConvLSTM import ConvLSTM
from src.models.DepthRouter import AdDepthRouted
from src.models.AdaptiveDepth import AdDepth, QLearningAgent


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="smoke_data/taxi_hourly.npy")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--with-rl", action="store_true",
                        help="also smoke-test the (fixed) Q-learning loop")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(0)
    np.random.seed(0)

    device = get_device()
    print(f"== device: {device} ==")

    # --- 1. Data: 24h of history as 6 stacks of 4h, predict next 4h ---------
    data = np.load(args.data)  # (T, H, W)
    frame_number, window = 4, 6
    train_loader, val_loader, test_loader = create_dataloaders(
        data, frame_number=frame_number, window_size=window,
        batch_size=32, train_split=0.7, val_split=0.15)
    xb, yb = next(iter(train_loader))
    print(f"data {data.shape} -> X {tuple(xb.shape)}, y {tuple(yb.shape)} | "
          f"train/val/test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    channels = frame_number
    tensor_shape = data.shape[-2:]
    hidden_dim = [channels]
    kernel_size = (3, 3)
    min_depth, max_depth = 1, 6

    def make_routed():
        return AdDepthRouted(channels, hidden_dim, kernel_size, len(hidden_dim),
                             channels, channels, tensor_shape, window,
                             max_depth=max_depth, min_depth=min_depth).to(device)

    # --- 2. Forward/backward sanity ----------------------------------------
    model = make_routed()
    n_params = sum(p.numel() for p in model.parameters())
    out = model(xb.float().to(device))
    assert out.shape == yb.shape, f"output {tuple(out.shape)} != target {tuple(yb.shape)}"
    loss = nn.functional.mse_loss(out, yb.float().to(device)) + 0.01 * model.depth_budget()
    loss.backward()
    router_grads = [p.grad.abs().sum().item() for p in model.router.parameters()]
    assert sum(router_grads) > 0, "router received no gradient"
    ed = model.expected_depth()
    assert ed.shape == (xb.shape[0], window)
    print(f"forward/backward OK | params: {n_params:,} | "
          f"initial E[depth] mean: {ed.mean().item():.2f}")

    # --- 3. Short routed training -------------------------------------------
    criterion = nn.MSELoss()
    model = make_routed()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    model, tr, vl, best_val, depth_profile, _ = train_model_routed(
        model, criterion, optimizer, train_loader, val_loader, test_loader,
        num_epochs=args.epochs, validation_interval=20, patience=15,
        logging=logging, device=device, lambda_budget=0.01,
        tau_start=2.0, tau_end=0.5, print_losses=False,
        checkpoint_path="/tmp/smoke_routed.pt")
    print(f"\n[AdDepthRouted] {time.time()-t0:.0f}s | first val {vl[0][1]:.4f} -> "
          f"best val {best_val:.4f} | depth profile {[round(d,1) for d in depth_profile]}")
    assert best_val < vl[0][1], "validation loss did not improve"

    # --- 4. ConvLSTM baseline at equal scale --------------------------------
    base = ConvLSTM(channels, hidden_dim, kernel_size, len(hidden_dim),
                    channels, channels, tensor_shape, window).to(device)
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-3)
    t0 = time.time()
    base, tr_b, vl_b, best_val_b, _ = train_model(
        base, criterion, optimizer, train_loader, val_loader, test_loader,
        num_epochs=args.epochs, validation_interval=20, patience=15,
        logging=logging, device=device, print_losses=False,
        checkpoint_path="/tmp/smoke_convlstm.pt")
    print(f"[ConvLSTM ] {time.time()-t0:.0f}s | first val {vl_b[0][1]:.4f} -> "
          f"best val {best_val_b:.4f}")

    # --- 5. Optional: bug-fixed Q-learning loop ------------------------------
    if args.with_rl:
        rl_model = AdDepth(channels, hidden_dim, kernel_size, len(hidden_dim),
                           channels, channels, tensor_shape, window,
                           initial_layers=max_depth).to(device)
        agent = QLearningAgent(window, init_loss=1.0)
        optimizer = torch.optim.AdamW(rl_model.parameters(), lr=1e-3)
        t0 = time.time()
        rl_model, _, vl_rl, best_val_rl, best_layers, _ = train_model_with_RL(
            rl_model, criterion, optimizer, train_loader, test_loader, val_loader,
            num_epochs=max(3, args.epochs // 2), validation_interval=20, patience=15,
            q_learning_agent=agent, window_size=window, max_depth=max_depth,
            initial_depth=3, logging=logging, device=device, min_depth=min_depth,
            Rl_start_step=2, loss_spike_threshold=1.5, print_losses=False,
            checkpoint_path="/tmp/smoke_rl.pt")
        print(f"[AdDepth-RL] {time.time()-t0:.0f}s | best val {best_val_rl:.4f} | "
              f"best layers {best_layers}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
