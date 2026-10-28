"""Build a small spatio-temporal taxi-demand tensor from raw TLC parquet files.

Produces smoke_data/taxi_hourly.npy of shape (T, H, W): hourly pickup counts
mapped onto the Manhattan zone grid (NYC_taxi_zoneMap.csv), log1p-scaled.
Intended for the Phase 1 smoke test, not for paper experiments.

Usage:
    python prepare_taxi_smoke.py --data_dir /path/to/Data_Taxi --months 2022-01 2022-02 2022-03
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_zone_grid(zone_map_csv):
    """Grid of TLC zone IDs (NaN = cell outside Manhattan). Returns (grid, H, W)."""
    grid = pd.read_csv(zone_map_csv).to_numpy(dtype=float)
    return grid, grid.shape[0], grid.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../Data_Taxi")
    parser.add_argument("--months", nargs="+", default=["2022-01", "2022-02", "2022-03"])
    parser.add_argument("--out", default="smoke_data/taxi_hourly.npy")
    args = parser.parse_args()

    grid, H, W = load_zone_grid(os.path.join(args.data_dir, "NYC_taxi_zoneMap.csv"))

    # zone id -> list of grid cells (a zone can span several cells)
    zone_cells = {}
    for r in range(H):
        for c in range(W):
            z = grid[r, c]
            if not np.isnan(z):
                zone_cells.setdefault(int(z), []).append((r, c))

    frames = []
    for month in sorted(args.months):
        path = os.path.join(args.data_dir, "Yellow_Cap_NYC_2022", f"yellow_tripdata_{month}.parquet")
        if not os.path.exists(path):
            path = os.path.join(args.data_dir, f"yellow_tripdata_{month}.parquet")
        df = pd.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])

        # Guard against the stray out-of-month rows present in TLC files.
        df = df[df["tpep_pickup_datetime"].dt.strftime("%Y-%m") == month]
        df = df[df["PULocationID"].isin(zone_cells.keys())]

        df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
        counts = df.groupby(["hour", "PULocationID"]).size().unstack(fill_value=0)

        full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="h")
        counts = counts.reindex(full_index, fill_value=0)

        month_tensor = np.zeros((len(counts), H, W), dtype=np.float32)
        for zone, cells in zone_cells.items():
            if zone in counts.columns:
                col = counts[zone].to_numpy(dtype=np.float32)
                share = col / len(cells)  # split a zone's demand across its cells
                for (r, c) in cells:
                    month_tensor[:, r, c] = share
        frames.append(month_tensor)
        print(f"{month}: {len(counts)} hours, {int(df.shape[0])} trips")

    tensor = np.concatenate(frames, axis=0)
    tensor = np.log1p(tensor)  # compress heavy-tailed counts

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, tensor)
    print(f"Saved {args.out}: shape={tensor.shape}, "
          f"min={tensor.min():.2f}, max={tensor.max():.2f}, mean={tensor.mean():.3f}")


if __name__ == "__main__":
    main()
