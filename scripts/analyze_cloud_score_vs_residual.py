"""Correlate soft cloud score M with cloudy-vs-clean residual magnitude.

Standalone diagnostic — does NOT train or modify model/bridge/loss/configs.

Usage (pylrt):
  python scripts/analyze_cloud_score_vs_residual.py --gpu 4 --seasons winter,summer,fall,spring
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.datasets.sen12mscr_utils import resolve_season_tokens
from src.utils.cloud_score import compute_soft_cloud_score
from src.utils.image_utils import save_gray_png, save_rgb_png
from src.utils.io_utils import ensure_dir


BINS = [
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 0.8),
    (0.8, 1.0 + 1e-12),  # include 1.0
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    p.add_argument(
        "--seasons",
        type=str,
        default="winter,summer,fall,spring",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_batches",
        type=int,
        default=80,
        help="Max train+val batches to process (stable stats without full pass).",
    )
    p.add_argument(
        "--corr_sample_max",
        type=int,
        default=500_000,
        help="Max pixels kept for Pearson/Spearman (reservoir).",
    )
    p.add_argument(
        "--bin_sample_max",
        type=int,
        default=200_000,
        help="Max pixels kept per bin for median/p90.",
    )
    p.add_argument("--num_vis", type=int, default=8)
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/spatial_mr_residual_analysis",
    )
    p.add_argument("--split", type=str, default="train_val", choices=["train_val", "all"])
    return p.parse_args()


def bin_label(lo, hi):
    if hi > 1.0:
        return f"[{lo:.1f},1.0]"
    return f"[{lo:.1f},{hi:.1f})"


class BinAccum:
    def __init__(self, sample_max: int):
        self.count = 0
        self.sum_m = 0.0
        self.sum_d = 0.0
        self.sum_d2 = 0.0
        self.sample_max = sample_max
        self.d_samples = []  # list of 1d float32 arrays
        self._n_samples = 0

    def update(self, m: np.ndarray, d: np.ndarray):
        n = int(m.size)
        if n == 0:
            return
        self.count += n
        self.sum_m += float(m.sum())
        self.sum_d += float(d.sum())
        self.sum_d2 += float(np.square(d, dtype=np.float64).sum())
        # reservoir-style: keep first sample_max, then random replace lightly
        remain = self.sample_max - self._n_samples
        if remain > 0:
            take = min(remain, n)
            self.d_samples.append(d[:take].astype(np.float32, copy=False))
            self._n_samples += take
            if take < n and self._n_samples >= self.sample_max:
                # fill remaining slots randomly from leftover
                leftover = d[take:]
                k = min(leftover.size, max(0, self.sample_max - self._n_samples))
                if k > 0:
                    idx = np.random.choice(leftover.size, size=k, replace=False)
                    self.d_samples.append(leftover[idx].astype(np.float32, copy=False))
                    self._n_samples += k
        else:
            # occasional random replacements to avoid first-batch bias
            k = min(n, max(1, n // 20))
            idx = np.random.choice(n, size=k, replace=False)
            # replace into concatenated buffer is expensive; just keep extra capped chunk
            if self._n_samples < self.sample_max * 1.05:
                self.d_samples.append(d[idx].astype(np.float32, copy=False))
                self._n_samples += k

    def finalize(self, total_pixels: int) -> dict:
        if self.count == 0:
            return {
                "pixel_count": 0,
                "fraction": 0.0,
                "mean_M": float("nan"),
                "mean_D": float("nan"),
                "median_D": float("nan"),
                "std_D": float("nan"),
                "p90_D": float("nan"),
            }
        mean_d = self.sum_d / self.count
        var = max(self.sum_d2 / self.count - mean_d * mean_d, 0.0)
        if self.d_samples:
            arr = np.concatenate(self.d_samples, axis=0)
            if arr.size > self.sample_max:
                arr = arr[: self.sample_max]
            median_d = float(np.median(arr))
            p90 = float(np.percentile(arr, 90))
        else:
            median_d = float("nan")
            p90 = float("nan")
        return {
            "pixel_count": int(self.count),
            "fraction": float(self.count / max(total_pixels, 1)),
            "mean_M": float(self.sum_m / self.count),
            "mean_D": float(mean_d),
            "median_D": median_d,
            "std_D": float(np.sqrt(var)),
            "p90_D": p90,
        }


class CorrReservoir:
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.m = []
        self.d = []
        self.n = 0

    def update(self, m: np.ndarray, d: np.ndarray):
        n = int(m.size)
        if n == 0:
            return
        remain = self.max_n - self.n
        if remain > 0:
            take = min(remain, n)
            self.m.append(m[:take].astype(np.float32, copy=False))
            self.d.append(d[:take].astype(np.float32, copy=False))
            self.n += take
            if take < n:
                leftover_m = m[take:]
                leftover_d = d[take:]
                k = min(leftover_m.size, self.max_n - self.n)
                if k > 0:
                    idx = np.random.choice(leftover_m.size, size=k, replace=False)
                    self.m.append(leftover_m[idx].astype(np.float32, copy=False))
                    self.d.append(leftover_d[idx].astype(np.float32, copy=False))
                    self.n += k
        else:
            # reservoir replacement
            k = min(n, max(1, n // 50))
            idx = np.random.choice(n, size=k, replace=False)
            # replace random positions in first buffer chunk if present
            if self.m:
                buf_m = np.concatenate(self.m)
                buf_d = np.concatenate(self.d)
                self.m = [buf_m]
                self.d = [buf_d]
                rep = np.random.choice(buf_m.size, size=k, replace=False)
                buf_m[rep] = m[idx]
                buf_d[rep] = d[idx]

    def arrays(self):
        if not self.m:
            return np.array([]), np.array([])
        return np.concatenate(self.m), np.concatenate(self.d)


def pearson_spearman(m: np.ndarray, d: np.ndarray):
    from scipy.stats import pearsonr, spearmanr

    if m.size < 3:
        return float("nan"), float("nan")
    # subsample further if huge
    if m.size > 300_000:
        idx = np.random.choice(m.size, size=300_000, replace=False)
        m = m[idx]
        d = d[idx]
    pr = pearsonr(m, d)
    sr = spearmanr(m, d)
    return float(pr.statistic if hasattr(pr, "statistic") else pr[0]), float(
        sr.statistic if hasattr(sr, "statistic") else sr[0]
    )


def make_accumulators(bin_sample_max: int):
    return {bin_label(lo, hi): BinAccum(bin_sample_max) for lo, hi in BINS}


def update_bins(accums, m_flat: np.ndarray, d_flat: np.ndarray):
    for (lo, hi), key in zip(BINS, [bin_label(lo, hi) for lo, hi in BINS]):
        if hi > 1.0:
            mask = (m_flat >= lo) & (m_flat <= 1.0)
        else:
            mask = (m_flat >= lo) & (m_flat < hi)
        if not np.any(mask):
            continue
        accums[key].update(m_flat[mask], d_flat[mask])


def rows_from_accums(scope: str, accums, total_pixels: int, pearson, spearman):
    rows = []
    for lo, hi in BINS:
        key = bin_label(lo, hi)
        stats = accums[key].finalize(total_pixels)
        rows.append(
            {
                "scope": scope,
                "bin": key,
                **stats,
                "pearson_M_D": pearson,
                "spearman_M_D": spearman,
            }
        )
    return rows


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    seasons = resolve_season_tokens(args.seasons)
    dataset = SEN12MSCRDataset(args.data_dir, seasons, return_index=True)
    print(f"dataset size={len(dataset)} seasons={seasons}")
    if len(dataset) == 0:
        print("No samples.")
        return

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.split == "train_val":
        # Concatenate indices from train+val
        indices = list(train_ds.indices) + list(val_ds.indices)
        eval_ds = Subset(dataset, indices)
        print(f"Using train+val: {len(eval_ds)} samples (seed={args.seed})")
    else:
        eval_ds = dataset
        print(f"Using all: {len(eval_ds)} samples")

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    global_acc = make_accumulators(args.bin_sample_max)
    global_corr = CorrReservoir(args.corr_sample_max)
    global_pixels = 0

    season_acc = {}
    season_corr = {}
    season_pixels = {}

    vis_dir = os.path.join(args.out_dir, "images")
    ensure_dir(vis_dir)
    saved = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.max_batches:
                break
            y, z, x0, idxs = batch
            y = y.to(device, non_blocking=True)
            x0 = x0.to(device, non_blocking=True)

            M = compute_soft_cloud_score(y)  # [B,1,H,W]
            D = torch.mean(torch.abs(y - x0), dim=1, keepdim=True)

            m_np = M.detach().cpu().numpy().reshape(-1)
            d_np = D.detach().cpu().numpy().reshape(-1)
            global_pixels += int(m_np.size)
            update_bins(global_acc, m_np, d_np)
            global_corr.update(m_np, d_np)

            # per-season via sample_meta
            idxs_list = idxs.tolist() if torch.is_tensor(idxs) else list(idxs)
            for j, ds_idx in enumerate(idxs_list):
                meta = dataset.sample_meta[int(ds_idx)]
                season = meta.get("season", "unknown")
                if season not in season_acc:
                    season_acc[season] = make_accumulators(args.bin_sample_max)
                    season_corr[season] = CorrReservoir(args.corr_sample_max // 4)
                    season_pixels[season] = 0
                mj = M[j].detach().cpu().numpy().reshape(-1)
                dj = D[j].detach().cpu().numpy().reshape(-1)
                season_pixels[season] += int(mj.size)
                update_bins(season_acc[season], mj, dj)
                season_corr[season].update(mj, dj)

            if bi % 10 == 0:
                print(
                    f"batch {bi}/{min(args.max_batches, len(loader))}: "
                    f"pixels={global_pixels:,} M_mean={float(M.mean()):.4f} "
                    f"D_mean={float(D.mean()):.4f}"
                )

            for j in range(y.size(0)):
                if saved >= args.num_vis:
                    break
                prefix = os.path.join(vis_dir, f"{saved:02d}")
                save_rgb_png(f"{prefix}_cloudy_rgb.png", y[j].cpu())
                save_rgb_png(f"{prefix}_clean_rgb.png", x0[j].cpu())
                save_gray_png(f"{prefix}_cloud_M.png", M[j].cpu())
                # residual display scaled by batch max for visibility
                d_vis = D[j].cpu()
                d_max = float(d_vis.max().clamp(min=1e-6))
                save_gray_png(f"{prefix}_residual_D.png", d_vis, vmin=0.0, vmax=d_max)
                saved += 1

    # correlations
    gm, gd = global_corr.arrays()
    pearson, spearman = pearson_spearman(gm, gd)
    print()
    print("=" * 72)
    print("GLOBAL (train+val)")
    print(f"pixels={global_pixels:,}  corr_samples={gm.size:,}")
    print(f"Pearson(M,D)={pearson:.6f}  Spearman(M,D)={spearman:.6f}")
    print(
        f"{'bin':<12} {'count':>12} {'frac':>8} {'meanM':>8} {'meanD':>8} "
        f"{'medD':>8} {'stdD':>8} {'p90D':>8}"
    )
    all_rows = rows_from_accums("global", global_acc, global_pixels, pearson, spearman)
    for row in all_rows:
        print(
            f"{row['bin']:<12} {row['pixel_count']:12d} {row['fraction']:8.4f} "
            f"{row['mean_M']:8.4f} {row['mean_D']:8.4f} {row['median_D']:8.4f} "
            f"{row['std_D']:8.4f} {row['p90_D']:8.4f}"
        )

    for season in sorted(season_acc.keys()):
        sm, sd = season_corr[season].arrays()
        p, s = pearson_spearman(sm, sd)
        print()
        print(f"SEASON={season} pixels={season_pixels[season]:,} Pearson={p:.6f} Spearman={s:.6f}")
        print(
            f"{'bin':<12} {'count':>12} {'frac':>8} {'meanM':>8} {'meanD':>8} "
            f"{'medD':>8} {'stdD':>8} {'p90D':>8}"
        )
        season_rows = rows_from_accums(
            f"season:{season}", season_acc[season], season_pixels[season], p, s
        )
        all_rows.extend(season_rows)
        for row in season_rows:
            print(
                f"{row['bin']:<12} {row['pixel_count']:12d} {row['fraction']:8.4f} "
                f"{row['mean_M']:8.4f} {row['mean_D']:8.4f} {row['median_D']:8.4f} "
                f"{row['std_D']:8.4f} {row['p90_D']:8.4f}"
            )

    out_csv = os.path.join(args.out_dir, "residual_by_cloud_score.csv")
    ensure_dir(args.out_dir)
    fieldnames = [
        "scope",
        "bin",
        "pixel_count",
        "fraction",
        "mean_M",
        "mean_D",
        "median_D",
        "std_D",
        "p90_D",
        "pearson_M_D",
        "spearman_M_D",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print()
    print(f"Wrote {out_csv}")
    print(f"Saved {saved} diagnostic image sets under {vis_dir}")
    print("No training was performed.")


if __name__ == "__main__":
    main()
