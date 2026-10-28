"""SOTA-scale comparable GATE: standard Moving MNIST 10->10 with a scaled ConvLSTM.

Goal of THIS script: confirm we can reproduce ~ConvLSTM-level accuracy on the
canonical 10-in/10-out Moving MNIST benchmark with a credible-scale backbone,
reported in the OpenSTL metric convention. This is the prerequisite gate before
investing in adding the positional depth schedule at this scale.

OpenSTL reference (10->10, MSE per-frame summed over 64x64 px, [0,1], lower better):
  ConvLSTM-S 29.8 (15M) | PredRNN 24.0 | PredRNN++ 22.1 | TAU 24.6 | SimVP 26.7.
Our evaluate_mse is PER-PIXEL mean; multiply by H*W (4096) to compare to the above.

Setup: X = frames[0:10] as window=10 stacks of 1 channel; y = frames[10:20] (10
frames) via out_channels=10. Split 8000/1000/1000 of mnist_test_seq.npy.
"""
import argparse, json, logging, sys, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from src.models.ConvLSTM import ConvLSTM
from src.train_loop import train_model, evaluate_mse


class MM10(Dataset):
    def __init__(self, data, in_frames, out_frames):
        self.data = data; self.i = in_frames; self.o = out_frames
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, k):
        seq = self.data[k]                       # (T, H, W) float32
        H, W = seq.shape[1], seq.shape[2]
        X = seq[:self.i].reshape(self.i, 1, H, W)        # window=in_frames, ch=1
        y = seq[self.i:self.i + self.o]                  # (out_frames, H, W)
        return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", default="../mnist_test_seq.npy")
    p.add_argument("--in_frames", type=int, default=10)
    p.add_argument("--out_frames", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="results/bench_mm10_convlstm.json")
    a = p.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True,
                        format=f"%(asctime)s [mm10|h{a.hidden}x{a.layers}|s{a.seed}] %(message)s")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    arr = np.transpose(np.load(a.npy), (1, 0, 2, 3)).astype(np.float32) / 255.0  # (N,T,H,W)
    ds = MM10(arr, a.in_frames, a.out_frames)
    dl = dict(num_workers=a.num_workers, pin_memory=True,
              persistent_workers=(a.num_workers > 0), prefetch_factor=(4 if a.num_workers else None))
    tr = DataLoader(Subset(ds, range(0, 8000)), batch_size=a.batch_size, shuffle=True, **dl)
    va = DataLoader(Subset(ds, range(8000, 9000)), batch_size=a.batch_size, shuffle=False, **dl)
    te = DataLoader(Subset(ds, range(9000, 10000)), batch_size=a.batch_size, shuffle=False, **dl)
    xb, yb = next(iter(tr))
    logging.info(f"X {tuple(xb.shape)} y {tuple(yb.shape)} | train/val/test 8000/1000/1000")

    model = ConvLSTM(input_dim=1, hidden_dim=[a.hidden] * a.layers, kernel_size=(3, 3),
                     num_layers=a.layers, in_channels=1, out_channels=a.out_frames,
                     tensor_shape=(64, 64), window=a.in_frames).to(dev)
    nparams = sum(x.numel() for x in model.parameters())
    logging.info(f"ConvLSTM params: {nparams/1e6:.2f}M  (OpenSTL ConvLSTM-S ~15M)")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    t0 = time.time()
    model, _, _, best_val, _ = train_model(
        model, nn.MSELoss(), opt, tr, va, te, a.epochs, a.val_interval, a.patience,
        logging, dev, print_losses=False, checkpoint_path=f"checkpoints/ckpt_mm10_s{a.seed}.pt", use_amp=a.amp)
    mse_px = evaluate_mse(model, te, dev)        # per-pixel mean MSE on [0,1]
    mse_openstl = mse_px * 64 * 64               # per-frame summed (OpenSTL convention)
    res = {"test_mse_perpixel": mse_px, "test_mse_openstl": mse_openstl,
           "best_val": best_val, "params_M": nparams / 1e6, "minutes": (time.time()-t0)/60,
           "hidden": a.hidden, "layers": a.layers, "in_frames": a.in_frames, "out_frames": a.out_frames}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n=== 10->10 ConvLSTM (h{a.hidden}x{a.layers}, {nparams/1e6:.2f}M) ===")
    print(f"  per-pixel MSE {mse_px:.5f} | OpenSTL-convention MSE {mse_openstl:.2f} "
          f"(ref ConvLSTM-S 29.8) | {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
