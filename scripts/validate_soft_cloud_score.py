"""Validate soft Sentinel-2 cloud score (no training changes).

Usage (from repo root, pylrt env):
  python scripts/validate_soft_cloud_score.py --gpu 4 --seasons winter --num_batches 4
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.datasets.sen12mscr_utils import BAND_ORDER, resolve_season_tokens
from src.utils.cloud_score import (
    S2_BAND_ORDER,
    cloud_score_summary,
    compute_soft_cloud_score,
)
from src.utils.io_utils import ensure_dir
from src.utils.image_utils import save_gray_png, save_rgb_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    p.add_argument("--seasons", type=str, default="winter")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--subset_max", type=int, default=64)
    p.add_argument("--num_vis", type=int, default=10)
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/soft_cloud_score_validation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def synthetic_unit_tests(device: torch.device) -> None:
    print("=" * 64)
    print("Unit tests (synthetic)")
    print("=" * 64)
    print(f"Repo BAND_ORDER: {list(BAND_ORDER)}")
    print(f"Cloud-score S2_BAND_ORDER: {list(S2_BAND_ORDER)}")
    assert tuple(BAND_ORDER) == S2_BAND_ORDER

    B, H, W = 4, 32, 32
    x = torch.rand(B, 13, H, W, device=device)

    # 1. Shape
    score = compute_soft_cloud_score(x)
    assert score.shape == (B, 1, H, W), score.shape
    print(f"[PASS] shape: {tuple(x.shape)} -> {tuple(score.shape)}")

    # 2. Range + finite
    assert float(score.min()) >= 0.0
    assert float(score.max()) <= 1.0
    assert torch.isfinite(score).all()
    print(
        f"[PASS] range: min={float(score.min()):.6f} max={float(score.max()):.6f} finite=True"
    )

    # 3. Determinism
    score2 = compute_soft_cloud_score(x)
    assert torch.equal(score, score2)
    print("[PASS] determinism: identical input -> identical score")

    # 4. Batch independence
    stacked = []
    for i in range(B):
        stacked.append(compute_soft_cloud_score(x[i : i + 1]))
    stacked = torch.cat(stacked, dim=0)
    max_abs = float((score - stacked).abs().max())
    assert max_abs < 1e-6, max_abs
    print(f"[PASS] batch independence: max_abs_diff={max_abs:.3e}")

    # Non-learnable / no grad
    x_req = x.clone().requires_grad_(True)
    s = compute_soft_cloud_score(x_req)
    assert not s.requires_grad
    print("[PASS] detached / non-learnable output")
    print()


def save_heatmap_png(path: str, score_hw: torch.Tensor) -> bool:
    """Simple blue->yellow->red heatmap for display only."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return save_gray_png(path, score_hw)

    ensure_dir(os.path.dirname(path))
    arr = score_hw.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    # piecewise RGB map
    r = np.clip(1.5 * arr - 0.2, 0.0, 1.0)
    g = np.clip(1.0 - 2.0 * np.abs(arr - 0.5), 0.0, 1.0)
    b = np.clip(1.2 - 1.5 * arr, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)
    return True


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    synthetic_unit_tests(device)

    seasons = resolve_season_tokens(args.seasons)
    print("=" * 64)
    print("Dataset validation / visualization")
    print("=" * 64)
    print(f"seasons={seasons}")
    ds = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    print(f"dataset size={len(ds)}")
    if len(ds) == 0:
        print("No samples; unit tests only.")
        return

    n = min(len(ds), args.subset_max) if args.subset_max > 0 else len(ds)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    subset = Subset(ds, idx)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    vis_dir = os.path.join(args.out_dir, "images")
    ensure_dir(vis_dir)
    stats_acc = []
    saved = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches and saved >= args.num_vis:
                break
            y = batch[0].to(device)  # cloudy
            score = compute_soft_cloud_score(y)
            stats_acc.append(score.cpu())

            if bi < args.num_batches:
                s = cloud_score_summary(score)
                print(
                    f"batch {bi}: mean={s['mean']:.4f} std={s['std']:.4f} "
                    f"min={s['min']:.4f} max={s['max']:.4f} "
                    f">0.2={s['frac_gt_0.2']:.3f} >0.5={s['frac_gt_0.5']:.3f} "
                    f">0.8={s['frac_gt_0.8']:.3f}"
                )

            for i in range(y.size(0)):
                if saved >= args.num_vis:
                    break
                # Display-only: RGB = B4,B3,B2 (indices 3,2,1); no model data change.
                save_rgb_png(
                    os.path.join(vis_dir, f"{saved:02d}_cloudy_rgb.png"),
                    y[i].cpu(),
                    rgb_indices=(3, 2, 1),
                )
                save_gray_png(
                    os.path.join(vis_dir, f"{saved:02d}_cloud_score_gray.png"),
                    score[i].cpu(),
                )
                save_heatmap_png(
                    os.path.join(vis_dir, f"{saved:02d}_cloud_score_heat.png"),
                    score[i].cpu(),
                )
                saved += 1

            if bi + 1 >= args.num_batches and saved >= args.num_vis:
                break

    if stats_acc:
        all_scores = torch.cat(stats_acc, dim=0)
        total = cloud_score_summary(all_scores)
        print()
        print("Aggregate over collected batches:")
        for k, v in total.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    print()
    print(f"Saved {saved} visualization triplets under: {vis_dir}")
    print("Validation complete. Training pipeline was NOT modified.")


if __name__ == "__main__":
    main()
