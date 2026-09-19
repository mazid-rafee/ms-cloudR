#!/usr/bin/env python3
"""Data-only spectral degradation diagnostic for SEN12MS-CR cloudy vs clean S2.

Reuses SEN12MSCRDataset preprocessing and the MR_r3 seed-42 80/10/10 split.
Does not train, load a model, touch the bridge, use SAR, or analyze the test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.datasets.sen12mscr_utils import BAND_ORDER
from src.utils.io_utils import map_seasons
from src.utils.sar_intervention import unwrap_subset

NUM_BANDS = len(BAND_ORDER)
EPS = 1e-8
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "spectral_bridge_diagnostic")


class BandAccumulators:
    """Float64 running sums for per-band cloudy/clean pixel statistics."""

    def __init__(self, num_bands: int = NUM_BANDS):
        self.num_bands = num_bands
        z = np.zeros(num_bands, dtype=np.float64)
        self.n = np.zeros(num_bands, dtype=np.float64)
        self.sum_y = z.copy()
        self.sum_x = z.copy()
        self.sum_y2 = z.copy()
        self.sum_x2 = z.copy()
        self.sum_xy = z.copy()
        self.sum_abs_diff = z.copy()
        self.sum_diff = z.copy()
        self.sum_diff2 = z.copy()

    def update(self, y: torch.Tensor, x0: torch.Tensor) -> None:
        """Accumulate one sample or batch. y, x0: [..., C, H, W] or [C, H, W]."""
        if y.ndim == 3:
            y = y.unsqueeze(0)
            x0 = x0.unsqueeze(0)
        if y.ndim != 4 or x0.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got y={tuple(y.shape)} x0={tuple(x0.shape)}")
        if y.shape[1] != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, got {y.shape[1]}")

        y64 = y.detach().cpu().to(torch.float64)
        x64 = x0.detach().cpu().to(torch.float64)
        # Flatten spatial+batch per band: [B, C, H, W] -> [C, N]
        y_flat = y64.permute(1, 0, 2, 3).reshape(self.num_bands, -1).numpy()
        x_flat = x64.permute(1, 0, 2, 3).reshape(self.num_bands, -1).numpy()
        diff = y_flat - x_flat
        n = float(y_flat.shape[1])

        self.n += n
        self.sum_y += y_flat.sum(axis=1)
        self.sum_x += x_flat.sum(axis=1)
        self.sum_y2 += np.square(y_flat).sum(axis=1)
        self.sum_x2 += np.square(x_flat).sum(axis=1)
        self.sum_xy += (y_flat * x_flat).sum(axis=1)
        self.sum_abs_diff += np.abs(diff).sum(axis=1)
        self.sum_diff += diff.sum(axis=1)
        self.sum_diff2 += np.square(diff).sum(axis=1)

    def finalize(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for b in range(self.num_bands):
            n = float(self.n[b])
            if n <= 0:
                rows.append(_empty_row(b, BAND_ORDER[b]))
                continue

            mean_y = self.sum_y[b] / n
            mean_x = self.sum_x[b] / n
            mean_diff = self.sum_diff[b] / n
            mae = self.sum_abs_diff[b] / n
            mse = self.sum_diff2[b] / n
            rmse = float(np.sqrt(max(mse, 0.0)))

            var_y = self.sum_y2[b] / n - mean_y * mean_y
            var_x = self.sum_x2[b] / n - mean_x * mean_x
            var_diff = self.sum_diff2[b] / n - mean_diff * mean_diff
            var_y = max(float(var_y), 0.0)
            var_x = max(float(var_x), 0.0)
            var_diff = max(float(var_diff), 0.0)
            std_y = float(np.sqrt(var_y))
            std_x = float(np.sqrt(var_x))
            std_diff = float(np.sqrt(var_diff))

            cov_xy = self.sum_xy[b] / n - mean_y * mean_x
            denom = std_y * std_x
            if denom <= EPS:
                pearson = float("nan")
            else:
                pearson = float(cov_xy / denom)
                if pearson > 1.0:
                    pearson = 1.0
                elif pearson < -1.0:
                    pearson = -1.0

            relative = float(mae / (std_x + EPS))

            rows.append(
                {
                    "band_index": b,
                    "band_name": BAND_ORDER[b],
                    "pixel_count": int(n),
                    "mae": float(mae),
                    "rmse": rmse,
                    "mean_signed_difference": float(mean_diff),
                    "std_difference": std_diff,
                    "pearson_correlation": pearson,
                    "clean_mean": float(mean_x),
                    "clean_std": std_x,
                    "cloudy_mean": float(mean_y),
                    "cloudy_std": std_y,
                    "relative_degradation": relative,
                }
            )
        return rows


def _empty_row(band_index: int, band_name: str) -> dict[str, Any]:
    return {
        "band_index": band_index,
        "band_name": band_name,
        "pixel_count": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "mean_signed_difference": float("nan"),
        "std_difference": float("nan"),
        "pearson_correlation": float("nan"),
        "clean_mean": float("nan"),
        "clean_std": float("nan"),
        "cloudy_mean": float("nan"),
        "cloudy_std": float("nan"),
        "relative_degradation": float("nan"),
    }


CSV_COLUMNS = [
    "band_index",
    "band_name",
    "pixel_count",
    "mae",
    "rmse",
    "mean_signed_difference",
    "std_difference",
    "pearson_correlation",
    "clean_mean",
    "clean_std",
    "cloudy_mean",
    "cloudy_std",
    "relative_degradation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-band cloudy-vs-clean spectral degradation diagnostic (train/val only)."
    )
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument(
        "--seasons",
        type=str,
        default="winter,summer,fall,spring",
        help="Comma-separated season tokens (MR_r3 order).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, use the first N indices of the seeded train Subset.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="If set, use the first N indices of the seeded val Subset.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Print progress every N samples (approx; batch-aware).",
    )
    return parser.parse_args()


def _cap_subset(subset: Subset, max_samples: Optional[int]) -> Subset:
    if max_samples is None:
        return subset
    if max_samples < 0:
        raise ValueError(f"max_samples must be >= 0 or None, got {max_samples}")
    n = min(int(max_samples), len(subset))
    # First N of the already-seeded Subset indices (deterministic; no reshuffle).
    capped_indices = list(subset.indices[:n])
    return Subset(subset.dataset, capped_indices)


def _split_sha256(root: SEN12MSCRDataset, indices: list[int]) -> str:
    h = hashlib.sha256()
    for idx in indices:
        s2c, s1, s2 = root.samples[idx]
        line = f"{idx}\t{s2c}\t{s1}\t{s2}\n"
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _plot_metric(
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["band_name"] for r in rows]
    values = [r[metric_key] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    ax.bar(x, values, color="#3d5a80", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_plots(rows: list[dict[str, Any]], prefix: str, output_dir: str) -> list[str]:
    specs = [
        ("mae", "MAE", f"{prefix} MAE by band (cloudy vs clean)"),
        ("rmse", "RMSE", f"{prefix} RMSE by band (cloudy vs clean)"),
        (
            "relative_degradation",
            "Relative degradation",
            f"{prefix} relative degradation by band (MAE / clean_std)",
        ),
        (
            "pearson_correlation",
            "Pearson correlation",
            f"{prefix} cloudy–clean Pearson correlation by band",
        ),
    ]
    name_map = {
        "mae": f"{prefix}_mae_by_band.png",
        "rmse": f"{prefix}_rmse_by_band.png",
        "relative_degradation": f"{prefix}_relative_degradation_by_band.png",
        "pearson_correlation": f"{prefix}_cloudy_clean_correlation_by_band.png",
    }
    paths = []
    for key, ylabel, title in specs:
        path = os.path.join(output_dir, name_map[key])
        _plot_metric(rows, key, title, ylabel, path)
        paths.append(path)
    return paths


def accumulate_split(
    loader: DataLoader,
    *,
    split_name: str,
    log_every: int,
) -> tuple[BandAccumulators, int, float]:
    acc = BandAccumulators()
    n_samples = 0
    t0 = time.perf_counter()
    total = len(loader.dataset)
    for batch in loader:
        y, _z, x0 = batch
        acc.update(y, x0)
        n_samples += int(y.shape[0])
        if log_every > 0 and (
            n_samples % log_every < y.shape[0] or n_samples >= total
        ):
            elapsed = time.perf_counter() - t0
            rate = n_samples / elapsed if elapsed > 0 else 0.0
            print(
                f"[{split_name}] {n_samples}/{total} samples "
                f"({100.0 * n_samples / max(total, 1):.1f}%) "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} samp/s",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    return acc, n_samples, elapsed


def _band_extremum(rows: list[dict[str, Any]], key: str, *, highest: bool) -> dict[str, Any]:
    valid = [r for r in rows if r.get(key) is not None and not _is_bad(r[key])]
    if not valid:
        return {"band_name": None, "value": None}
    best = max(valid, key=lambda r: r[key]) if highest else min(valid, key=lambda r: r[key])
    return {"band_name": best["band_name"], "band_index": best["band_index"], "value": best[key]}


def _is_bad(v: Any) -> bool:
    try:
        return not np.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def _fmt_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'band':<5} {'MAE':>10} {'RMSE':>10} {'rel_deg':>10} "
        f"{'pearson':>10} {'clean_std':>10} {'signed':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['band_name']:<5} "
            f"{r['mae']:10.6f} "
            f"{r['rmse']:10.6f} "
            f"{r['relative_degradation']:10.6f} "
            f"{r['pearson_correlation']:10.6f} "
            f"{r['clean_std']:10.6f} "
            f"{r['mean_signed_difference']:10.6f}"
        )


def _train_val_comparison(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    print("\n=== Train vs Validation comparison (per band) ===")
    header = (
        f"{'band':<5} {'tr_MAE':>10} {'va_MAE':>10} {'dMAE':>10} "
        f"{'tr_rel':>10} {'va_rel':>10} {'d_rel':>10} "
        f"{'tr_r':>8} {'va_r':>8}"
    )
    print(header)
    print("-" * len(header))
    for tr, va in zip(train_rows, val_rows):
        d_mae = tr["mae"] - va["mae"]
        d_rel = tr["relative_degradation"] - va["relative_degradation"]
        print(
            f"{tr['band_name']:<5} "
            f"{tr['mae']:10.6f} {va['mae']:10.6f} {d_mae:10.6f} "
            f"{tr['relative_degradation']:10.6f} {va['relative_degradation']:10.6f} {d_rel:10.6f} "
            f"{tr['pearson_correlation']:8.4f} {va['pearson_correlation']:8.4f}"
        )


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    seasons = map_seasons(args.seasons)
    print("Building SEN12MSCRDataset (filesystem enumeration order preserved)...")
    print(f"  data_dir={args.data_dir}")
    print(f"  seasons={seasons}")
    print(f"  seed={args.seed}")

    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    full_size = len(dataset)
    n_ignored = len(getattr(dataset, "ignore_set", set()))
    print(f"  full_dataset_size={full_size} (ignored keys from invalid_files.txt: {n_ignored})")
    if full_size == 0:
        print("ERROR: no samples found.", file=sys.stderr)
        return 1

    train_size = int(0.8 * full_size)
    val_size = int(0.1 * full_size)
    test_size = full_size - train_size - val_size
    train_ds, val_ds, _test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Never touch test_ds beyond size accounting.
    del _test_ds

    train_root, train_indices_full = unwrap_subset(train_ds)
    val_root, val_indices_full = unwrap_subset(val_ds)
    assert train_root is dataset and val_root is dataset

    train_ds_use = _cap_subset(train_ds, args.max_train_samples)
    val_ds_use = _cap_subset(val_ds, args.max_val_samples)
    _, train_indices = unwrap_subset(train_ds_use)
    _, val_indices = unwrap_subset(val_ds_use)

    train_sha = _split_sha256(dataset, train_indices)
    val_sha = _split_sha256(dataset, val_indices)

    split_audit = {
        "seed": args.seed,
        "seasons": seasons,
        "seasons_cli": args.seasons,
        "full_dataset_size": full_size,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "invalid_samples_excluded": n_ignored,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "train_samples_analyzed": len(train_indices),
        "val_samples_analyzed": len(val_indices),
        "first_20_train_root_indices": train_indices[:20],
        "first_20_val_root_indices": val_indices[:20],
        "train_split_sha256": train_sha,
        "val_split_sha256": val_sha,
        "note": (
            "SHA256 digests cover the ordered analyzed indices only "
            "(capped if max_*_samples is set). Paths hashed as "
            "idx\\tcloudy\\tSAR\\tclean per sample."
        ),
    }
    split_audit_path = os.path.join(args.output_dir, "split_audit.json")
    _write_json(split_audit_path, split_audit)
    print(f"Wrote {split_audit_path}")
    print(f"  train_split_sha256={train_sha}")
    print(f"  val_split_sha256={val_sha}")

    train_loader = DataLoader(
        train_ds_use,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds_use,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(
        f"\nAnalyzing TRAIN: {len(train_indices)} samples "
        f"(of {train_size} in full train split)..."
    )
    train_acc, n_train, train_elapsed = accumulate_split(
        train_loader, split_name="train", log_every=args.log_every
    )
    train_rows = train_acc.finalize()

    print(
        f"\nAnalyzing VAL: {len(val_indices)} samples "
        f"(of {val_size} in full val split)..."
    )
    val_acc, n_val, val_elapsed = accumulate_split(
        val_loader, split_name="val", log_every=max(args.log_every // 5, 1)
    )
    val_rows = val_acc.finalize()

    train_csv = os.path.join(args.output_dir, "train_band_statistics.csv")
    val_csv = os.path.join(args.output_dir, "val_band_statistics.csv")
    _write_csv(train_csv, train_rows)
    _write_csv(val_csv, val_rows)

    plot_paths = []
    plot_paths.extend(_save_plots(train_rows, "train", args.output_dir))
    plot_paths.extend(_save_plots(val_rows, "val", args.output_dir))

    def _nan_inf_report(rows: list[dict[str, Any]], split: str) -> list[str]:
        issues = []
        for r in rows:
            for k, v in r.items():
                if k in ("band_index", "band_name", "pixel_count"):
                    continue
                if _is_bad(v):
                    issues.append(f"{split}/{r['band_name']}/{k}={v}")
        return issues

    issues = _nan_inf_report(train_rows, "train") + _nan_inf_report(val_rows, "val")

    summary = {
        "config": {
            "data_dir": args.data_dir,
            "seasons": seasons,
            "seasons_cli": args.seasons,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "output_dir": args.output_dir,
            "band_order": list(BAND_ORDER),
            "normalization": "SEN12MSCRDataset: clamp[0,10000]/10000 (no re-normalization)",
            "split": "random_split 80/10/10; test unused",
            "diagnostic_scope": "data-only cloudy y vs clean x0; no SAR, no model, no bridge",
        },
        "counts": {
            "full_dataset_size": full_size,
            "train_split_size": train_size,
            "val_split_size": val_size,
            "test_split_size": test_size,
            "train_samples_analyzed": n_train,
            "val_samples_analyzed": n_val,
            "invalid_samples_excluded": n_ignored,
        },
        "runtime_seconds": {
            "train": train_elapsed,
            "val": val_elapsed,
            "total": train_elapsed + val_elapsed,
        },
        "split_hashes": {
            "train_split_sha256": train_sha,
            "val_split_sha256": val_sha,
        },
        "train_band_statistics": train_rows,
        "val_band_statistics": val_rows,
        "comparisons": {
            "highest_mae_band": _band_extremum(train_rows, "mae", highest=True),
            "lowest_mae_band": _band_extremum(train_rows, "mae", highest=False),
            "highest_relative_degradation_band": _band_extremum(
                train_rows, "relative_degradation", highest=True
            ),
            "lowest_relative_degradation_band": _band_extremum(
                train_rows, "relative_degradation", highest=False
            ),
            "highest_cloudy_clean_correlation_band": _band_extremum(
                train_rows, "pearson_correlation", highest=True
            ),
            "lowest_cloudy_clean_correlation_band": _band_extremum(
                train_rows, "pearson_correlation", highest=False
            ),
        },
        "nan_inf_issues": issues,
        "disclaimer": (
            "This diagnostic only measures whether cloudy-to-clean degradation "
            "differs across Sentinel-2 bands. It does not recommend r_b values "
            "and does not validate SpectralMR."
        ),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    _write_json(summary_path, summary)

    # Terminal tables
    _fmt_table(train_rows, "TRAIN statistics (canonical band order)")
    train_by_rel = sorted(train_rows, key=lambda r: r["relative_degradation"], reverse=True)
    _fmt_table(train_by_rel, "TRAIN sorted by relative degradation (desc)")
    train_by_mae = sorted(train_rows, key=lambda r: r["mae"], reverse=True)
    _fmt_table(train_by_mae, "TRAIN sorted by MAE (desc)")

    _fmt_table(val_rows, "VAL statistics (canonical band order)")
    _train_val_comparison(train_rows, val_rows)

    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")
    print(f"Train runtime: {train_elapsed:.1f}s | Val runtime: {val_elapsed:.1f}s")
    if issues:
        print(f"WARNING: {len(issues)} NaN/Inf value(s) detected:")
        for item in issues[:20]:
            print(f"  {item}")
    else:
        print("No NaN/Inf issues in computed band statistics.")

    created = [
        split_audit_path,
        train_csv,
        val_csv,
        summary_path,
        *plot_paths,
    ]
    print("\nFiles created:")
    for p in created:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
