"""Deterministic soft Sentinel-2 cloud score (DSen2-CR-style).

Operates on already-normalized cloudy S2 tensors in [0, 1]
(after clip[0,10000]/10000). No trainable parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Confirmed by src/datasets/sen12mscr_utils.BAND_ORDER and
# data/SEN12MS-CR/sen12ms_cr_dataLoader.py S2Bands.ALL (rasterio file order).
S2_BAND_ORDER = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
)
IDX_B1 = 0
IDX_B2 = 1
IDX_B3 = 2
IDX_B4 = 3
IDX_B10 = 10
IDX_B11 = 11


def rescale(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Linear rescale; high may be < low (intentional for snow rejection)."""
    return (x - low) / (high - low)


def _min_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Erosion via min-pooling (= -max_pool(-x))."""
    pad = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)


def _max_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def morphological_closing(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Closing ≈ dilation then erosion on an odd neighborhood."""
    return _min_pool2d(_max_pool2d(x, kernel_size), kernel_size)


def local_average(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


@torch.no_grad()
def compute_soft_cloud_score(cloudy_s2: torch.Tensor) -> torch.Tensor:
    """Compute soft cloud probability/score from cloudy Sentinel-2.

    Args:
        cloudy_s2: [B, 13, H, W] already in [0, 1] (do not /10000 again).

    Returns:
        cloud_score: [B, 1, H, W] in [0, 1], detached / non-learnable.
    """
    if cloudy_s2.ndim != 4 or cloudy_s2.size(1) != 13:
        raise ValueError(
            f"expected cloudy_s2 shape [B,13,H,W], got {tuple(cloudy_s2.shape)}"
        )

    x = cloudy_s2.detach().float()
    b1 = x[:, IDX_B1 : IDX_B1 + 1]
    b2 = x[:, IDX_B2 : IDX_B2 + 1]
    b3 = x[:, IDX_B3 : IDX_B3 + 1]
    b4 = x[:, IDX_B4 : IDX_B4 + 1]
    b10 = x[:, IDX_B10 : IDX_B10 + 1]
    b11 = x[:, IDX_B11 : IDX_B11 + 1]

    score = torch.ones_like(b1)

    # Brightness tests (DSen2-CR).
    score = torch.minimum(score, rescale(b2, 0.1, 0.5))
    score = torch.minimum(score, rescale(b1, 0.1, 0.3))
    score = torch.minimum(score, rescale(b1 + b10, 0.5, 0.7))
    score = torch.minimum(score, rescale(b2 + b3 + b4, 0.2, 0.8))

    # Snow rejection: high NDSI lowers score (bounds 0.8 -> 0.6 intentional).
    ndsi = (b3 - b11) / (b3 + b11 + 1e-6)
    score = torch.minimum(score, rescale(ndsi, 0.8, 0.6))

    # DSen2-CR-style smoothing: ~5x5 closing then 7x7 local mean.
    score = morphological_closing(score, kernel_size=5)
    score = local_average(score, kernel_size=7)
    score = torch.clamp(score, 0.0, 1.0)

    return score.detach()


def cloud_score_summary(cloud_score: torch.Tensor) -> dict:
    """Aggregate statistics for one or more score maps [B,1,H,W] or flattened."""
    s = cloud_score.detach().float().reshape(-1)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(unbiased=False)),
        "min": float(s.min()),
        "max": float(s.max()),
        "frac_gt_0.2": float((s > 0.2).float().mean()),
        "frac_gt_0.5": float((s > 0.5).float().mean()),
        "frac_gt_0.8": float((s > 0.8).float().mean()),
        "numel": int(s.numel()),
    }
