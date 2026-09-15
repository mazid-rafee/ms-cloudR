#!/usr/bin/env python3
"""Measure processed SAR range and how often unbounded Gaussian noise exits it.

Does not change noise generation. Inference/training are not run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torch.utils.data import Subset, random_split

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.utils.io_utils import map_seasons
from src.utils.sar_intervention import noise_sar_like, unwrap_subset


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_out_of_range_prob(mu: float, sigma: float, lo: float = 0.0, hi: float = 1.0) -> dict:
    if sigma <= 0:
        below = 1.0 if mu < lo else 0.0
        above = 1.0 if mu > hi else 0.0
        return {"below": below, "above": above, "total": below + above}
    below = _norm_cdf((lo - mu) / sigma)
    above = 1.0 - _norm_cdf((hi - mu) / sigma)
    return {"below": below, "above": above, "total": below + above}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument("--seasons", type=str, default="winter,summer,fall,spring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sar_intervention_seed", type=int, default=123)
    parser.add_argument("--subset_max", type=int, default=0)
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=512,
        help="Cap test-set SAR reads for this diagnostic (0 = all test samples).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/DBCR_MR_r3_SAR_intervention/sar_noise_range_audit.json",
    )
    args = parser.parse_args()

    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=map_seasons(args.seasons))
    if args.subset_max > 0:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(dataset), generator=g)[: args.subset_max]
        dataset = Subset(dataset, idx.tolist())
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    _, _, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    root, test_indices = unwrap_subset(test_ds)
    if args.max_test_samples > 0:
        test_indices = test_indices[: args.max_test_samples]

    mins = torch.tensor([float("inf"), float("inf")])
    maxs = torch.tensor([float("-inf"), float("-inf")])
    sum_c = torch.zeros(2, dtype=torch.float64)
    sumsq_c = torch.zeros(2, dtype=torch.float64)
    count = 0
    n_at_floor = torch.zeros(2, dtype=torch.int64)
    n_at_ceil = torch.zeros(2, dtype=torch.int64)

    for i, ds_idx in enumerate(test_indices):
        _, z, _ = root[int(ds_idx)]
        z2 = z[:2]
        mins = torch.minimum(mins, z2.amin(dim=(1, 2)))
        maxs = torch.maximum(maxs, z2.amax(dim=(1, 2)))
        flat = z2.reshape(2, -1).to(torch.float64)
        sum_c += flat.sum(dim=1)
        sumsq_c += (flat * flat).sum(dim=1)
        n_at_floor += (z2 <= 0.0).reshape(2, -1).sum(dim=1)
        n_at_ceil += (z2 >= 1.0).reshape(2, -1).sum(dim=1)
        count += int(flat.shape[1])
        if (i + 1) % 50 == 0 or i + 1 == len(test_indices):
            print(f"scanned {i + 1}/{len(test_indices)} test SAR samples", flush=True)

    mean = sum_c / count
    std = torch.sqrt(torch.clamp(sumsq_c / count - mean * mean, min=0.0))
    stats = {
        "source": "processed_test_set_sar",
        "n_samples": len(test_indices),
        "n_test_split": test_size,
        "n_pixels_per_channel": count,
        "theoretical_normalized_range": [0.0, 1.0],
        "preprocessing": (
            "VV: clamp([-25,0]); (VV+25)/25. "
            "VH: clamp([-32.5,0]); (VH+32.5)/32.5."
        ),
        "channels": {
            "VV": {
                "min": float(mins[0]),
                "max": float(maxs[0]),
                "mean": float(mean[0]),
                "std": float(std[0]),
                "frac_at_0": float(n_at_floor[0] / count),
                "frac_at_1": float(n_at_ceil[0] / count),
            },
            "VH": {
                "min": float(mins[1]),
                "max": float(maxs[1]),
                "mean": float(mean[1]),
                "std": float(std[1]),
                "frac_at_0": float(n_at_floor[1] / count),
                "frac_at_1": float(n_at_ceil[1] / count),
            },
        },
    }

    noise_stats = {
        "channels": {
            "VV": {"mean": float(mean[0]), "std": float(std[0])},
            "VH": {"mean": float(mean[1]), "std": float(std[1])},
        }
    }
    n_below = torch.zeros(2, dtype=torch.int64)
    n_above = torch.zeros(2, dtype=torch.int64)
    n_noise = 0
    dummy = torch.zeros(2, 256, 256)
    for i in range(len(test_indices)):
        z_n = noise_sar_like(dummy, noise_stats, args.sar_intervention_seed, i)
        n_below += (z_n < 0.0).reshape(2, -1).sum(dim=1)
        n_above += (z_n > 1.0).reshape(2, -1).sum(dim=1)
        n_noise += z_n.shape[1] * z_n.shape[2]

    empirical = {
        "VV": {
            "frac_below_0": float(n_below[0] / n_noise),
            "frac_above_1": float(n_above[0] / n_noise),
            "frac_outside_0_1": float((n_below[0] + n_above[0]) / n_noise),
        },
        "VH": {
            "frac_below_0": float(n_below[1] / n_noise),
            "frac_above_1": float(n_above[1] / n_noise),
            "frac_outside_0_1": float((n_below[1] + n_above[1]) / n_noise),
        },
        "both_channels_frac_outside_0_1": float((n_below.sum() + n_above.sum()) / (2 * n_noise)),
        "n_generated_pixels_per_channel": n_noise,
    }
    theoretical = {
        "VV": gaussian_out_of_range_prob(float(mean[0]), float(std[0])),
        "VH": gaussian_out_of_range_prob(float(mean[1]), float(std[1])),
    }

    payload = {
        "observed_processed_sar": stats,
        "unbounded_gaussian_noise": {
            "empirical": empirical,
            "theoretical_normal_cdf": theoretical,
            "clipped": False,
            "recommendation": (
                "A non-trivial fraction of N(mu,sigma) pixels fall outside [0,1]. "
                "Clipping to [0,1] would keep the intervention on the same manifold "
                "as real processed SAR. Leaving it unbounded is a harsher OOD probe. "
                "Do not change until approved."
            ),
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
