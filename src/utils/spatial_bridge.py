"""Spatially adaptive mean-reverting DB-CR training bridge.

Preserves scalar MR_r3 (s = t / T) and adds SpatialMR_r3:

    M = soft cloud score in [0, 1]
    r_map = r_max * M
    A(s) = (1 - exp(-r_map * s)) / (1 - exp(-r_map))
         → s when r_map → 0
    x_t = x0 + A * (y - x0)

No trainable parameters. Cloud score is detached; x_t is not.
"""

from __future__ import annotations

import torch

from src.models.dbcr import mean_reverting_alpha_schedule
from src.utils.cloud_score import compute_soft_cloud_score

SPATIAL_SCHEDULE_NAMES = {"spatial_mr_r3", "spatial_mean_reverting"}
SCALAR_MR_NAMES = {"mean_reverting", "mr_r3"}


def normalize_bridge_schedule(name: str) -> str:
    n = str(name).lower().replace("-", "_")
    if n in {"original", "sinusoidal"}:
        return "original"
    if n in SCALAR_MR_NAMES:
        return "mean_reverting"
    if n in SPATIAL_SCHEDULE_NAMES:
        return "spatial_mr_r3"
    return n


def is_spatial_mr_schedule(name: str) -> bool:
    return normalize_bridge_schedule(name) == "spatial_mr_r3"


def _normalized_time(t, T) -> torch.Tensor:
    """Exact MR_r3 convention: s = t / float(T)."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    return t.float() / float(T)


def spatial_mr_alpha(t, T, cloud_score_m: torch.Tensor, r_max: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Spatially adaptive MR alpha map A ∈ [B,1,H,W].

    Args:
        t: scalar or [B] timesteps (same convention as mean_reverting_alpha_schedule).
        T: total diffusion steps.
        cloud_score_m: [B,1,H,W] soft cloud score in [0,1] (detached).
        r_max: maximum mean-reversion rate (MR_r3 uses 3.0).
    """
    if cloud_score_m.ndim != 4 or cloud_score_m.size(1) != 1:
        raise ValueError(
            f"cloud_score_m must be [B,1,H,W], got {tuple(cloud_score_m.shape)}"
        )

    m = cloud_score_m.detach().float()
    s = _normalized_time(t, T)
    if s.ndim == 0:
        s_b = s.view(1, 1, 1, 1).expand(m.size(0), 1, m.size(2), m.size(3))
    elif s.ndim == 1:
        if s.numel() == 1:
            s_b = s.view(1, 1, 1, 1).expand(m.size(0), 1, m.size(2), m.size(3))
        else:
            if int(s.numel()) != int(m.size(0)):
                raise ValueError(
                    f"t batch {int(s.numel())} != cloud_score batch {int(m.size(0))}"
                )
            s_b = s.view(-1, 1, 1, 1).expand(-1, 1, m.size(2), m.size(3))
    else:
        s_b = s.reshape(m.size(0), 1, 1, 1).expand(-1, 1, m.size(2), m.size(3))

    s_b = s_b.to(device=m.device, dtype=m.dtype)
    r_map = float(r_max) * m

    # A = (1 - exp(-r s)) / (1 - exp(-r)) via expm1; limit A=s as r→0.
    numerator = -torch.expm1(-r_map * s_b)
    denominator = -torch.expm1(-r_map)
    A = torch.where(r_map.abs() < eps, s_b, numerator / denominator)
    return A


def construct_spatial_mr_bridge(
    x0: torch.Tensor,
    y: torch.Tensor,
    t,
    T: int,
    r_max: float = 3.0,
    cloud_score_m: torch.Tensor | None = None,
):
    """Build SpatialMR_r3 state x_t = x0 + A * (y - x0).

    Returns:
        x_t: [B,13,H,W] (gradients flow through x0/y path; A uses detached M)
        A: [B,1,H,W]
        M: [B,1,H,W] cloud score
        r_map: [B,1,H,W]
    """
    if cloud_score_m is None:
        cloud_score_m = compute_soft_cloud_score(y)
    A = spatial_mr_alpha(t, T, cloud_score_m, r_max=r_max)
    r_map = float(r_max) * cloud_score_m.detach().float()
    x_t = x0 + A * (y - x0)
    return x_t, A, cloud_score_m, r_map


def construct_scalar_mr_bridge(x0, y, t, T, rate: float = 3.0):
    """Existing DBCR_MR_r3 bridge for comparison / regression tests."""
    alpha = mean_reverting_alpha_schedule(t, T, rate=rate)
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=x0.dtype, device=x0.device)
    alpha = alpha.view(-1, 1, 1, 1).to(dtype=x0.dtype, device=x0.device)
    if alpha.size(0) == 1 and x0.size(0) > 1:
        alpha = alpha.expand(x0.size(0), 1, 1, 1)
    x_t = (1.0 - alpha) * x0 + alpha * y
    return x_t, alpha
