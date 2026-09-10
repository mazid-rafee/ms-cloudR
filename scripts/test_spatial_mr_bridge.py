"""Mathematical unit tests for SpatialMR_r3 training bridge."""

from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dbcr import mean_reverting_alpha_schedule
from src.utils.spatial_bridge import (
    construct_scalar_mr_bridge,
    construct_spatial_mr_bridge,
    spatial_mr_alpha,
)

TOL = 1e-5
T = 1000
R_MAX = 3.0


def _assert_close(a, b, tol=TOL, msg=""):
    diff = float((a - b).abs().max())
    assert diff <= tol, f"{msg} max_abs={diff} > {tol}"


def test_all_cloud_equivalence(device):
    print("A. All-cloud equivalence (M=1 ↔ scalar MR_r3)")
    B, H, W = 3, 16, 16
    x0 = torch.rand(B, 13, H, W, device=device)
    y = torch.rand(B, 13, H, W, device=device)
    M = torch.ones(B, 1, H, W, device=device)
    ts = torch.tensor([0, 250, 500, 750, 1000], device=device, dtype=torch.float32)
    for t_scalar in ts:
        t = t_scalar.repeat(B)
        A = spatial_mr_alpha(t, T, M, r_max=R_MAX)
        alpha = mean_reverting_alpha_schedule(t, T, rate=R_MAX).view(-1, 1, 1, 1)
        _assert_close(A, alpha.expand_as(A), msg=f"alpha @ t={int(t_scalar)}")
        x_sp, _, _, _ = construct_spatial_mr_bridge(x0, y, t, T, r_max=R_MAX, cloud_score_m=M)
        x_sc, _ = construct_scalar_mr_bridge(x0, y, t, T, rate=R_MAX)
        _assert_close(x_sp, x_sc, msg=f"x_t @ t={int(t_scalar)}")
    print("  [PASS]")


def test_all_clear_limit(device):
    print("B. All-clear limit (M=0 → A=s)")
    B, H, W = 2, 8, 8
    x0 = torch.rand(B, 13, H, W, device=device)
    y = torch.rand(B, 13, H, W, device=device)
    M = torch.zeros(B, 1, H, W, device=device)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = torch.full((B,), frac * T, device=device)
        A = spatial_mr_alpha(t, T, M, r_max=R_MAX)
        s = torch.full_like(A, frac)
        _assert_close(A, s, msg=f"A==s @ s={frac}")
        x_t, _, _, _ = construct_spatial_mr_bridge(x0, y, t, T, r_max=R_MAX, cloud_score_m=M)
        expected = x0 + s * (y - x0)
        _assert_close(x_t, expected, msg=f"x_t @ s={frac}")
    print("  [PASS]")


def test_endpoints(device):
    print("C. Endpoint tests (random soft M)")
    B, H, W = 2, 12, 12
    x0 = torch.rand(B, 13, H, W, device=device)
    y = torch.rand(B, 13, H, W, device=device)
    M = torch.rand(B, 1, H, W, device=device)
    t0 = torch.zeros(B, device=device)
    t1 = torch.full((B,), float(T), device=device)
    A0 = spatial_mr_alpha(t0, T, M, r_max=R_MAX)
    A1 = spatial_mr_alpha(t1, T, M, r_max=R_MAX)
    _assert_close(A0, torch.zeros_like(A0), msg="A(s=0)")
    _assert_close(A1, torch.ones_like(A1), msg="A(s=1)")
    x0_t, _, _, _ = construct_spatial_mr_bridge(x0, y, t0, T, r_max=R_MAX, cloud_score_m=M)
    x1_t, _, _, _ = construct_spatial_mr_bridge(x0, y, t1, T, r_max=R_MAX, cloud_score_m=M)
    _assert_close(x0_t, x0, msg="x_t(s=0)=x0")
    _assert_close(x1_t, y, msg="x_t(s=1)=y")
    print("  [PASS]")


def test_range(device):
    print("D. Range / finite")
    B, H, W = 4, 16, 16
    M = torch.rand(B, 1, H, W, device=device)
    for frac in torch.linspace(0, 1, 11, device=device):
        t = torch.full((B,), float(frac * T), device=device)
        A = spatial_mr_alpha(t, T, M, r_max=R_MAX)
        assert torch.isfinite(A).all()
        assert float(A.min()) >= -1e-5
        assert float(A.max()) <= 1.0 + 1e-5
    print("  [PASS]")


def test_monotonicity(device):
    print("E. Monotonicity in time")
    B, H, W = 2, 10, 10
    M = torch.rand(B, 1, H, W, device=device)
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    As = []
    for frac in fracs:
        t = torch.full((B,), frac * T, device=device)
        As.append(spatial_mr_alpha(t, T, M, r_max=R_MAX))
    for i in range(len(As) - 1):
        # allow tiny FP noise
        assert torch.all(As[i + 1] + 1e-6 >= As[i]), f"non-monotone at i={i}"
    print("  [PASS]")


def test_known_values(device):
    print("F. Known values at s=0.5")
    B, H, W = 1, 4, 4
    t = torch.full((B,), 0.5 * T, device=device)
    # M=1 → ~0.817574
    A1 = spatial_mr_alpha(t, T, torch.ones(B, 1, H, W, device=device), r_max=R_MAX)
    alpha_ref = float(mean_reverting_alpha_schedule(t, T, rate=R_MAX)[0])
    _assert_close(A1, torch.full_like(A1, alpha_ref), tol=1e-5, msg="M=1 @0.5")
    assert abs(alpha_ref - 0.817574) < 1e-4, alpha_ref
    # M=0 → 0.5
    A0 = spatial_mr_alpha(t, T, torch.zeros(B, 1, H, W, device=device), r_max=R_MAX)
    _assert_close(A0, torch.full_like(A0, 0.5), msg="M=0 @0.5")
    # M=0.5 intermediate
    A05 = spatial_mr_alpha(t, T, torch.full((B, 1, H, W), 0.5, device=device), r_max=R_MAX)
    # r=1.5 → (1-exp(-1.5*0.5))/(1-exp(-1.5))
    r = 1.5
    expected = (1.0 - torch.exp(torch.tensor(-r * 0.5, device=device))) / (
        1.0 - torch.exp(torch.tensor(-r, device=device))
    )
    _assert_close(A05, torch.full_like(A05, float(expected)), tol=1e-5, msg="M=0.5 @0.5")
    print(f"  M=1 alpha≈{float(A1.mean()):.6f} (ref {alpha_ref:.6f})")
    print(f"  M=0 alpha={float(A0.mean()):.6f}")
    print(f"  M=0.5 alpha≈{float(A05.mean()):.6f}")
    print("  [PASS]")


def test_shared_spectral(device):
    print("G. Shared spectral schedule (A is [B,1,H,W])")
    B, H, W = 2, 8, 8
    M = torch.rand(B, 1, H, W, device=device)
    t = torch.randint(0, T + 1, (B,), device=device).float()
    A = spatial_mr_alpha(t, T, M, r_max=R_MAX)
    assert A.shape == (B, 1, H, W), A.shape
    x0 = torch.rand(B, 13, H, W, device=device)
    y = torch.rand(B, 13, H, W, device=device)
    x_t, A2, _, _ = construct_spatial_mr_bridge(x0, y, t, T, r_max=R_MAX, cloud_score_m=M)
    # All 13 bands share the same A multiplier:
    for c in range(13):
        expected_c = x0[:, c : c + 1] + A2 * (y[:, c : c + 1] - x0[:, c : c + 1])
        _assert_close(x_t[:, c : c + 1], expected_c, msg=f"band {c}")
    print("  [PASS]")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    test_all_cloud_equivalence(device)
    test_all_clear_limit(device)
    test_endpoints(device)
    test_range(device)
    test_monotonicity(device)
    test_known_values(device)
    test_shared_spectral(device)
    print()
    print("All SpatialMR_r3 mathematical unit tests passed.")


if __name__ == "__main__":
    main()
