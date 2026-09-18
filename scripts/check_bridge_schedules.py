"""Sanity-check DB-CR alpha schedules (original, scalar MR, spectral MR)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dbcr import (
    NUM_S2_BANDS,
    alpha_schedule,
    get_alpha_schedule,
    mean_reverting_alpha_schedule,
    reshape_alpha_for_broadcast,
)
from src.utils.config import load_config

TOL = 1e-6
ALL_R3 = [3.0] * NUM_S2_BANDS
GROUPED = [2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0]


def _format_row(values):
    return " ".join(f"{v:0.6f}" for v in values)


def test_scalar_regression():
    print("=" * 72)
    print("SCALAR MR_r3 REGRESSION")
    print("=" * 72)
    T = 1000
    rate = 3.0
    fractions = [0.00, 0.25, 0.50, 0.75, 1.00]
    t = torch.tensor([f * T for f in fractions], dtype=torch.float32)

    original = alpha_schedule(t, T)
    mean_rev = mean_reverting_alpha_schedule(t, T, rate=rate)
    via_factory = get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)(t, T)

    print("Original DB-CR alpha:")
    print("t/T:", " ".join(f"{f:0.2f}" for f in fractions))
    print("alpha:", _format_row(original.tolist()))
    print()
    print(f"Mean-reverting alpha (rate={rate}):")
    print("t/T:", " ".join(f"{f:0.2f}" for f in fractions))
    print("alpha:", _format_row(mean_rev.tolist()))
    print()

    assert abs(float(mean_rev[0])) < TOL, f"alpha(0) should be ~0, got {mean_rev[0]}"
    assert abs(float(mean_rev[-1]) - 1.0) < TOL, f"alpha(T) should be ~1, got {mean_rev[-1]}"
    for i in range(len(mean_rev) - 1):
        assert float(mean_rev[i + 1]) >= float(mean_rev[i]) - 1e-8, (
            f"mean-reverting alpha must be non-decreasing: "
            f"{mean_rev[i]} -> {mean_rev[i + 1]}"
        )
    assert torch.allclose(mean_rev, via_factory), "factory scalar path mismatch"

    t_cpu = t.cpu()
    out_cpu = mean_reverting_alpha_schedule(t_cpu, T, rate=rate)
    assert out_cpu.device == t_cpu.device

    if torch.cuda.is_available():
        try:
            t_dev = t.cuda()
            out_dev = mean_reverting_alpha_schedule(t_dev, T, rate=rate)
            assert out_dev.device == t_dev.device
            _ = out_dev.cpu()
        except RuntimeError as exc:
            print(f"CUDA device check skipped ({exc}).")

    try:
        mean_reverting_alpha_schedule(t, T, rate=0.0)
        raise AssertionError("rate<=0 should raise ValueError")
    except ValueError:
        pass

    # Config load without spectral field must yield scalar factory.
    cfg = load_config("configs/dbcr_mean_reverting.json")
    assert "spectral_mean_reversion_rates" not in cfg
    fn = get_alpha_schedule(
        bridge_schedule=cfg["bridge_schedule"],
        mean_reversion_rate=cfg["mean_reversion_rate"],
        spectral_mean_reversion_rates=cfg.get("spectral_mean_reversion_rates"),
    )
    assert torch.allclose(fn(t, T), mean_rev)

    print("SCALAR MR_r3 REGRESSION: PASS")
    print()


def test_shapes():
    print("=" * 72)
    print("SHAPE TESTS")
    print("=" * 72)
    T = 1000
    B = 4
    t_batch = torch.randint(0, T + 1, (B,)).float()
    t_scalar = torch.tensor(float(T // 2))

    a_bs = mean_reverting_alpha_schedule(t_batch, T, rate=3.0)
    a_ss = mean_reverting_alpha_schedule(t_scalar, T, rate=3.0)
    a_bv = mean_reverting_alpha_schedule(t_batch, T, rate=ALL_R3)
    a_sv = mean_reverting_alpha_schedule(t_scalar, T, rate=ALL_R3)

    assert a_bs.shape == (B,), a_bs.shape
    assert a_ss.ndim == 0, a_ss.shape
    assert a_bv.shape == (B, NUM_S2_BANDS), a_bv.shape
    assert a_sv.shape == (NUM_S2_BANDS,), a_sv.shape

    assert reshape_alpha_for_broadcast(a_bs).shape == (B, 1, 1, 1)
    assert reshape_alpha_for_broadcast(a_ss).shape == (1, 1, 1, 1)
    assert reshape_alpha_for_broadcast(a_bv).shape == (B, NUM_S2_BANDS, 1, 1)
    assert reshape_alpha_for_broadcast(a_sv, as_channels=True).shape == (
        1,
        NUM_S2_BANDS,
        1,
        1,
    )
    print("SHAPE TESTS: PASS")
    print()


def test_spectral_endpoints_mono_finite():
    print("=" * 72)
    print("SPECTRAL ENDPOINTS / MONOTONICITY / FINITE")
    print("=" * 72)
    T = 1000
    rates = GROUPED
    t_grid = torch.arange(0, T + 1, dtype=torch.float32)
    alpha = mean_reverting_alpha_schedule(t_grid, T, rate=rates)
    assert alpha.shape == (T + 1, NUM_S2_BANDS)

    assert torch.isfinite(alpha).all(), "NaN/Inf in spectral alpha grid"
    a0 = alpha[0]
    aT = alpha[-1]
    assert torch.max(torch.abs(a0)).item() < TOL, f"alpha(0)={a0.tolist()}"
    assert torch.max(torch.abs(aT - 1.0)).item() < TOL, f"alpha(T)={aT.tolist()}"

    diffs = alpha[1:] - alpha[:-1]
    assert torch.min(diffs).item() >= -1e-8, f"non-monotone min_diff={torch.min(diffs)}"
    print("SPECTRAL ENDPOINTS / MONOTONICITY / FINITE: PASS")
    print()


def test_all_r3_equivalence():
    print("=" * 72)
    print("ALL-r3 EQUIVALENCE (alpha)")
    print("=" * 72)
    T = 1000
    checkpoints = [0, 1, T // 4, T // 2, (3 * T) // 4, T]
    t_fixed = torch.tensor(checkpoints, dtype=torch.float32)
    scalar = mean_reverting_alpha_schedule(t_fixed, T, rate=3.0)
    spectral = mean_reverting_alpha_schedule(t_fixed, T, rate=ALL_R3)
    # spectral [N,13]; compare each band to scalar [N]
    diff_fixed = torch.max(torch.abs(spectral - scalar.unsqueeze(-1))).item()

    g = torch.Generator().manual_seed(42)
    t_rand = torch.randint(0, T + 1, (64,), generator=g).float()
    scalar_r = mean_reverting_alpha_schedule(t_rand, T, rate=3.0)
    spectral_r = mean_reverting_alpha_schedule(t_rand, T, rate=ALL_R3)
    diff_rand = torch.max(torch.abs(spectral_r - scalar_r.unsqueeze(-1))).item()

    max_abs = max(diff_fixed, diff_rand)
    print(f"max_abs_alpha_difference={max_abs:.6e}")
    assert max_abs < 1e-6, max_abs

    # Factory / config path
    cfg = load_config("configs/dbcr_spectral_mr_all_r3.json")
    fn = get_alpha_schedule(
        bridge_schedule=cfg["bridge_schedule"],
        mean_reversion_rate=cfg["mean_reversion_rate"],
        spectral_mean_reversion_rates=cfg["spectral_mean_reversion_rates"],
    )
    via_cfg = fn(t_fixed, T)
    assert torch.allclose(via_cfg, spectral, atol=1e-7)

    print("ALL-r3 EQUIVALENCE (alpha): PASS")
    print()
    return max_abs


def main():
    test_scalar_regression()
    test_shapes()
    test_spectral_endpoints_mono_finite()
    max_abs = test_all_r3_equivalence()
    print("All schedule sanity checks passed.")
    print(f"Reported max_abs_alpha_difference={max_abs:.6e}")


if __name__ == "__main__":
    main()
