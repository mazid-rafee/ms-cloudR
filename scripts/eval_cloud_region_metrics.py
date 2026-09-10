"""Region-wise reconstruction diagnostics by soft cloud score M (NFE=1).

Does not change the model or bridge. Loads a trained checkpoint and reports
MAE/L1 in low-cloud (M<0.1), high-cloud (M>0.8), and optional M bins.

Usage:
  python scripts/eval_cloud_region_metrics.py \\
    --gpu 0 \\
    --checkpoint outputs/DBCR_SpatialMR_r3_seed42_50_epochs/checkpoints/best.pt \\
    --bridge_schedule spatial_mr_r3 \\
    --run_name DBCR_SpatialMR_r3_seed42_50_epochs_region_eval
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.models.dbcr import get_alpha_schedule
from src.models.registry import get_model
from src.utils.checkpoint import load_checkpoint
from src.utils.cloud_score import compute_soft_cloud_score
from src.utils.io_utils import ensure_dir, map_seasons, save_json
from src.utils.logger import setup_logger


BINS = [
    ("low_cloud_M_lt_0.1", None, 0.1),  # M < 0.1
    ("high_cloud_M_gt_0.8", 0.8, None),  # M > 0.8
    ("bin_[0.0,0.1)", 0.0, 0.1),
    ("bin_[0.1,0.3)", 0.1, 0.3),
    ("bin_[0.3,0.5)", 0.3, 0.5),
    ("bin_[0.5,0.8)", 0.5, 0.8),
    ("bin_[0.8,1.0]", 0.8, 1.0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    p.add_argument("--seasons", type=str, default="winter,summer,fall,spring")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nfe", type=int, default=1)
    p.add_argument(
        "--bridge_schedule",
        type=str,
        default="spatial_mr_r3",
        choices=[
            "original",
            "mean_reverting",
            "mr_r3",
            "spatial_mr_r3",
            "spatial_mean_reverting",
        ],
    )
    p.add_argument("--mean_reversion_rate", type=float, default=3.0)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default="region_eval")
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def mask_for_bin(M, lo, hi, special=None):
    if special == "low":
        return M < 0.1
    if special == "high":
        return M > 0.8
    if hi >= 1.0 and lo >= 0.8:
        return (M >= lo) & (M <= 1.0)
    return (M >= lo) & (M < hi)


def main():
    args = parse_args()
    if int(args.nfe) != 1:
        raise SystemExit("This diagnostic is intended for NFE=1 only.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(args.output_dir, args.run_name)
    log_dir = os.path.join(run_dir, "logs")
    logger = setup_logger(log_dir, "region_eval")
    logger.info("device=%s checkpoint=%s", device, args.checkpoint)

    seasons = map_seasons(args.seasons)
    dataset = SEN12MSCRDataset(args.data_dir, seasons)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    _, _, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("test size=%d", len(test_ds))

    model = get_model("dbcr")().to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)
    model.eval()
    alpha_fn = get_alpha_schedule(args.bridge_schedule, args.mean_reversion_rate)

    # accumulators: abs_sum, count
    stats = {
        name: {"abs_sum": 0.0, "count": 0}
        for name, _, _ in [
            ("low_cloud_M_lt_0.1", None, None),
            ("high_cloud_M_gt_0.8", None, None),
            ("bin_[0.0,0.1)", None, None),
            ("bin_[0.1,0.3)", None, None),
            ("bin_[0.3,0.5)", None, None),
            ("bin_[0.5,0.8)", None, None),
            ("bin_[0.8,1.0]", None, None),
            ("all", None, None),
        ]
    }
    total_pixels = 0

    with torch.no_grad():
        for step, (y, z, x0) in enumerate(loader, start=1):
            y = y.to(device)
            z = z.to(device)
            x0 = x0.to(device)

            # NFE=1 inference (same as eval.py)
            x_t = y
            T = args.diffusion_steps
            t_curr = torch.tensor(T, device=device)
            t_next = torch.tensor(0, device=device)
            alpha_curr = alpha_fn(t_curr.float(), T).view(1, 1, 1, 1)
            alpha_next = alpha_fn(t_next.float(), T).view(1, 1, 1, 1)
            x0_hat = model(x_t, t_curr.repeat(x_t.size(0)), z)
            # NFE=1: collapses to x0_hat
            _ = (1 - alpha_next / alpha_curr) * x0_hat + (alpha_next / alpha_curr) * x_t

            err = torch.abs(x0_hat - x0)  # [B,13,H,W]
            # mean over bands for region MAE (consistent with residual diagnostic D)
            err_map = err.mean(dim=1, keepdim=True)  # [B,1,H,W]
            M = compute_soft_cloud_score(y)

            total_pixels += int(M.numel())
            stats["all"]["abs_sum"] += float(err_map.sum())
            stats["all"]["count"] += int(err_map.numel())

            masks = {
                "low_cloud_M_lt_0.1": M < 0.1,
                "high_cloud_M_gt_0.8": M > 0.8,
                "bin_[0.0,0.1)": (M >= 0.0) & (M < 0.1),
                "bin_[0.1,0.3)": (M >= 0.1) & (M < 0.3),
                "bin_[0.3,0.5)": (M >= 0.3) & (M < 0.5),
                "bin_[0.5,0.8)": (M >= 0.5) & (M < 0.8),
                "bin_[0.8,1.0]": (M >= 0.8) & (M <= 1.0),
            }
            for name, mask in masks.items():
                if mask.any():
                    stats[name]["abs_sum"] += float(err_map[mask].sum())
                    stats[name]["count"] += int(mask.sum().item())

            if step % args.log_every == 0 or step == len(loader):
                logger.info("eval step %d/%d", step, len(loader))

    rows = []
    for name, st in stats.items():
        count = st["count"]
        mae = st["abs_sum"] / count if count > 0 else float("nan")
        rows.append(
            {
                "region": name,
                "pixel_count": count,
                "fraction": count / max(total_pixels, 1),
                "mae_l1": mae,
            }
        )
        logger.info(
            "%s: mae=%.6f count=%d frac=%.4f",
            name,
            mae,
            count,
            count / max(total_pixels, 1),
        )

    ensure_dir(run_dir)
    csv_path = os.path.join(run_dir, "region_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["region", "pixel_count", "fraction", "mae_l1"]
        )
        w.writeheader()
        w.writerows(rows)
    save_json(
        os.path.join(run_dir, "region_metrics.json"),
        {
            "checkpoint": args.checkpoint,
            "bridge_schedule": args.bridge_schedule,
            "nfe": args.nfe,
            "seed": args.seed,
            "regions": rows,
        },
    )
    logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
