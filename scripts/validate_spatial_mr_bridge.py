"""Debug visualizations / stats for SpatialMR_r3 vs scalar MR_r3."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.datasets.sen12mscr_utils import resolve_season_tokens
from src.models.dbcr import mean_reverting_alpha_schedule
from src.utils.cloud_score import compute_soft_cloud_score
from src.utils.image_utils import save_gray_png, save_rgb_png
from src.utils.io_utils import ensure_dir
from src.utils.spatial_bridge import (
    construct_scalar_mr_bridge,
    construct_spatial_mr_bridge,
    spatial_mr_alpha,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    p.add_argument("--seasons", type=str, default="winter")
    p.add_argument("--subset_max", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=3)
    p.add_argument("--num_vis", type=int, default=6)
    p.add_argument("--r_max", type=float, default=3.0)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/spatial_mr_r3_validation",
    )
    return p.parse_args()


def save_heat(path, arr):
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return save_gray_png(path, arr)
    ensure_dir(os.path.dirname(path))
    a = arr.detach().cpu().float().numpy()
    if a.ndim == 3:
        a = a[0]
    a = a.clip(0, 1)
    r = (1.5 * a - 0.2).clip(0, 1)
    g = (1.0 - 2.0 * abs(a - 0.5)).clip(0, 1)
    b = (1.2 - 1.5 * a).clip(0, 1)
    Image.fromarray((np.stack([r, g, b], -1) * 255).astype("uint8")).save(path)
    return True


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.diffusion_steps
    r_max = args.r_max

    seasons = resolve_season_tokens(args.seasons)
    ds = SEN12MSCRDataset(args.data_dir, seasons)
    n = min(len(ds), args.subset_max) if args.subset_max > 0 else len(ds)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:n]
    loader = DataLoader(
        Subset(ds, idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    vis_dir = os.path.join(args.out_dir, "images")
    ensure_dir(vis_dir)
    saved = 0
    fracs = (0.25, 0.50, 0.75)

    print(f"device={device} r_max={r_max} T={T}")
    with torch.no_grad():
        for bi, (y, z, x0) in enumerate(loader):
            if bi >= args.num_batches and saved >= args.num_vis:
                break
            y = y.to(device)
            x0 = x0.to(device)
            M = compute_soft_cloud_score(y)
            r_map = r_max * M

            print(f"\nbatch {bi}:")
            print(
                f"  M: min={float(M.min()):.4f} max={float(M.max()):.4f} "
                f"mean={float(M.mean()):.4f} std={float(M.std()):.4f}"
            )
            print(
                f"  r_map: min={float(r_map.min()):.4f} max={float(r_map.max()):.4f} "
                f"mean={float(r_map.mean()):.4f}"
            )

            for frac in fracs:
                t = torch.full((y.size(0),), frac * T, device=device)
                A = spatial_mr_alpha(t, T, M, r_max=r_max)
                alpha_sc = mean_reverting_alpha_schedule(t, T, rate=r_max).view(-1, 1, 1, 1)
                x_sp, _, _, _ = construct_spatial_mr_bridge(
                    x0, y, t, T, r_max=r_max, cloud_score_m=M
                )
                x_sc, _ = construct_scalar_mr_bridge(x0, y, t, T, rate=r_max)
                mad_A = float((A - alpha_sc).abs().mean())
                mad_x = float((x_sp - x_sc).abs().mean())
                print(
                    f"  s={frac:.2f}: A mean/min/max="
                    f"{float(A.mean()):.4f}/{float(A.min()):.4f}/{float(A.max()):.4f} "
                    f"| scalar_alpha={float(alpha_sc.mean()):.4f} "
                    f"| mean|A-alpha|={mad_A:.4e} mean|x_sp-x_sc|={mad_x:.4e}"
                )

            for i in range(y.size(0)):
                if saved >= args.num_vis:
                    break
                prefix = os.path.join(vis_dir, f"{saved:02d}")
                save_rgb_png(f"{prefix}_1_cloudy_rgb.png", y[i].cpu())
                save_gray_png(f"{prefix}_2_cloud_M.png", M[i].cpu())
                save_heat(f"{prefix}_3_r_map.png", (r_map[i] / r_max).cpu())  # normalize display
                save_gray_png(f"{prefix}_3b_r_map_gray.png", (r_map[i] / r_max).cpu())

                for frac in fracs:
                    tag = f"{int(frac * 100):02d}"
                    t = torch.full((1,), frac * T, device=device)
                    Mi = M[i : i + 1]
                    A = spatial_mr_alpha(t, T, Mi, r_max=r_max)
                    x_sp, _, _, _ = construct_spatial_mr_bridge(
                        x0[i : i + 1], y[i : i + 1], t, T, r_max=r_max, cloud_score_m=Mi
                    )
                    x_sc, _ = construct_scalar_mr_bridge(
                        x0[i : i + 1], y[i : i + 1], t, T, rate=r_max
                    )
                    save_heat(f"{prefix}_4_A_s{tag}.png", A[0].cpu())
                    save_rgb_png(f"{prefix}_5_xt_spatial_s{tag}.png", x_sp[0].cpu())
                    save_rgb_png(f"{prefix}_6_xt_mr_r3_s{tag}.png", x_sc[0].cpu())
                saved += 1

    print(f"\nSaved visualizations under: {vis_dir}")
    print("Note: at s=1 both paths reach cloudy_s2 (not drawn).")


if __name__ == "__main__":
    main()
