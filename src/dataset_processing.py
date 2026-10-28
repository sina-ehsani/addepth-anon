import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py


class MovingMNISTDataset(Dataset):
    """Moving MNIST: independent sequences (N, T, H, W), values in [0, 1].

    Mirrors the (window, frame_number) stacking the models expect: for each
    sequence, X = the first window*frame_number frames as `window` non-
    overlapping stacks of `frame_number` frames, y = the next frame_number
    frames. Unlike WeatherPerceptionDataset there is NO sliding within a
    sequence — each sequence is one sample (sequences are i.i.d.).
    """

    def __init__(self, data, frame_number, window_size):
        self.data = data                       # (N, T, H, W) float32 in [0,1]
        self.frame_number = frame_number
        self.window_size = window_size
        self.need = window_size * frame_number + frame_number
        if data.shape[1] < self.need:
            raise ValueError(f"seq_len {data.shape[1]} < window*fn+fn = {self.need}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        seq = self.data[i]                     # (T, H, W)
        H, W = seq.shape[1], seq.shape[2]
        wfn = self.window_size * self.frame_number
        X = seq[:wfn].reshape(self.window_size, self.frame_number, H, W)
        y = seq[wfn: wfn + self.frame_number]
        return X, y


def create_dataloaders_mmnist(npy_path, frame_number, window_size, batch_size,
                              train_n=8000, val_n=1000, num_workers=8, pin_memory=False,
                              max_seqs=None):
    """Train/val/test loaders for Moving MNIST.

    The .npy is (seq_len, num_seqs, H, W) uint8; we transpose to (num_seqs,
    seq_len, H, W) and scale to [0,1]. Sequences are i.i.d., so the split is a
    fixed slice: first train_n train, next val_n val, the rest test.
    """
    arr = np.load(npy_path)                    # (T, N, H, W) uint8
    arr = np.transpose(arr, (1, 0, 2, 3)).astype(np.float32) / 255.0
    if max_seqs:
        arr = arr[:max_seqs]
    n = arr.shape[0]
    if train_n + val_n >= n:
        raise ValueError(f"train_n+val_n={train_n+val_n} >= num_seqs={n}")
    dataset = MovingMNISTDataset(arr, frame_number, window_size)
    train_idx = list(range(0, train_n))
    val_idx = list(range(train_n, train_n + val_n))
    test_idx = list(range(train_n + val_n, n))

    dl = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        dl.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True, drop_last=False, **dl)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, **dl)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, **dl)
    return train_loader, val_loader, test_loader


class WeatherPerceptionDataset(Dataset):
    def __init__(self, data, frame_number, window_size):
        """
        Initialize the dataset.

        :param data: The original dataset of shape (time_steps, height, width).
        :param frame_number: Number of frames in each DataStack.
        :param window_size: Number of DataStacks in each window.

        Each sample consists of:
          X: `window_size` consecutive, NON-overlapping stacks of `frame_number`
             frames each, i.e. frames [idx, idx + window_size * frame_number).
          y: the `frame_number` frames immediately AFTER the input window.
        The input and target never share frames (no target leakage).
        """
        self.data = data
        self.frame_number = frame_number
        self.window_size = window_size
        self.total_frames = data.shape[0]
        self.frames_per_datapoint = self.frame_number * self.window_size

    def __len__(self):
        # Each sample needs frames_per_datapoint input frames plus
        # frame_number target frames.
        return max(0, self.total_frames - self.frames_per_datapoint - self.frame_number + 1)

    def __getitem__(self, idx):
        if idx + self.frames_per_datapoint + self.frame_number > self.total_frames:
            raise IndexError("Index out of range")

        # X: (window_size, frame_number, height, width). The stacks are
        # consecutive and non-overlapping, so the whole input is one contiguous
        # block [idx : idx + window*frame_number] reshaped — no per-stack list
        # comprehension and no redundant copy (the DataLoader collate copies once
        # into the batch tensor). dtype follows self.data (float16 = half the
        # IPC/pin/H2D bandwidth, the bottleneck; upcast to fp32 on the GPU).
        H, W = self.data.shape[1], self.data.shape[2]
        wfn = self.frames_per_datapoint
        X = self.data[idx: idx + wfn].reshape(self.window_size, self.frame_number, H, W)

        # y: (frame_number, height, width) — frames strictly after the input.
        y = self.data[idx + wfn: idx + wfn + self.frame_number]

        return X, y


def create_dataloaders(data, frame_number, window_size, batch_size, train_split, val_split,
                       num_workers=0, pin_memory=False, stride=1):
    """
    Create chronological train/val/test DataLoaders.

    The split is CHRONOLOGICAL (not random): the earliest `train_split`
    fraction is used for training, the next `val_split` fraction for
    validation, and the most recent remainder for testing. A gap of
    frame_number * (window_size + 1) frames is left between consecutive
    splits so that no input or target frame is shared across splits
    (overlapping sliding windows would otherwise leak data).

    :param data: The original dataset of shape (time_steps, height, width).
    :param frame_number, window_size: Parameters for the dataset.
    :param batch_size: Batch size for the DataLoader.
    :param train_split, val_split: Split ratios for training and validation.
    """
    dataset = WeatherPerceptionDataset(data, frame_number, window_size)
    n = len(dataset)

    # Number of sample indices to drop between splits so windows don't
    # straddle a split boundary.
    gap = frame_number * (window_size + 1)

    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    # stride>1 subsamples sample indices: adjacent sliding windows overlap by
    # frame_number*window_size-1 frames (near-duplicates), so striding keeps the
    # full temporal/seasonal span at a fraction of the samples with ~no info loss.
    train_indices = list(range(0, max(0, train_end - gap), stride))
    val_indices = list(range(train_end, max(train_end, val_end - gap), stride))
    test_indices = list(range(val_end, n, stride))

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            f"Empty split: train={len(train_indices)}, val={len(val_indices)}, "
            f"test={len(test_indices)} (n={n}, gap={gap}). Each split must span "
            f"more than {gap} samples; increase the data size or adjust the ratios."
        )

    # Worker/pin settings shared across loaders. With num_workers>0 the
    # WeatherPerceptionDataset (which holds `data` in RAM) is shared
    # copy-on-write across forked workers — reads only, so no duplication.
    dl_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=4)

    # Shuffling within the training set is safe (samples are i.i.d. given the
    # chronological split); val/test are kept in temporal order.
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size,
                              shuffle=True, drop_last=False, **dl_kwargs)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size,
                            shuffle=False, **dl_kwargs)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size,
                             shuffle=False, **dl_kwargs)

    return train_loader, val_loader, test_loader


def create_dataloader_taxi(batch_size, h5_path="../all_tensors_7_zone2.h5",
                           val_fraction=0.5, pool_resplit=False):
    """
    Create train/val/test DataLoaders for the NYC taxi dataset.

    The holdout tensors stored in the H5 file are split CHRONOLOGICALLY into a
    validation set (first `val_fraction`) and a test set (the remainder).
    The test set is therefore never seen by early stopping or by the RL
    depth-adaptation loop, which both consume only the validation loss.

    :param batch_size: Batch size for the DataLoaders.
    :param h5_path: Path to the H5 file containing the preprocessed tensors.
    :param val_fraction: Fraction of the holdout period used for validation.
    """
    with h5py.File(h5_path, 'r') as hf:
        tensor_pu = hf['tensor_pu'][:]
        tensor_pu_ts = hf['tensor_pu_ts'][:]
        tensor_do = hf['tensor_do'][:]
        tensor_holdout_pu_ts = hf['tensor_test_pu_ts'][:]
        tensor_holdout_pu = hf['tensor_test_pu'][:]
        tensor_holdout_do = hf['tensor_test_do'][:]

    if pool_resplit:
        # Pool train(723)+holdout(144) chronologically and re-split 70/15/15 for a
        # bigger, more reliable test (~130 vs 72). Order = [train (earlier); holdout (later)].
        import numpy as np
        puts = np.concatenate([tensor_pu_ts, tensor_holdout_pu_ts], 0)
        do = np.concatenate([tensor_do, tensor_holdout_do], 0)
        pu = np.concatenate([tensor_pu, tensor_holdout_pu], 0)
        n = puts.shape[0]; tr, va = int(n * 0.70), int(n * 0.85)
        def mk(a, b, shuf):
            t = lambda z: torch.tensor(z, dtype=torch.float32)
            return DataLoader(torch.utils.data.TensorDataset(t(puts[a:b]), t(do[a:b]), t(pu[a:b])),
                              batch_size=batch_size, shuffle=shuf)
        return mk(0, tr, True), mk(tr, va, False), mk(va, n, False)

    train_do = torch.tensor(tensor_do, dtype=torch.float32)
    train_pu_ts = torch.tensor(tensor_pu_ts, dtype=torch.float32)
    train_labels = torch.tensor(tensor_pu, dtype=torch.float32)

    # Chronological val/test split of the holdout period.
    n_holdout = tensor_holdout_pu.shape[0]
    split = int(n_holdout * val_fraction)
    if split == 0 or split == n_holdout:
        raise ValueError(
            f"val_fraction={val_fraction} leaves an empty validation or test set "
            f"(holdout size {n_holdout})."
        )

    val_do = torch.tensor(tensor_holdout_do[:split], dtype=torch.float32)
    val_pu_ts = torch.tensor(tensor_holdout_pu_ts[:split], dtype=torch.float32)
    val_labels = torch.tensor(tensor_holdout_pu[:split], dtype=torch.float32)

    test_do = torch.tensor(tensor_holdout_do[split:], dtype=torch.float32)
    test_pu_ts = torch.tensor(tensor_holdout_pu_ts[split:], dtype=torch.float32)
    test_labels = torch.tensor(tensor_holdout_pu[split:], dtype=torch.float32)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_pu_ts, train_do, train_labels),
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(val_pu_ts, val_do, val_labels),
        batch_size=batch_size,
        shuffle=False)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_pu_ts, test_do, test_labels),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, val_loader, test_loader
