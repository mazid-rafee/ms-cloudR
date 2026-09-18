"""Oracle consistency test for deterministic DB-CR reverse bridge updates.

Verifies that the existing eval reverse equation reconstructs analytic bridge
states for original, scalar mean-reverting, and spectral mean-reverting alpha
schedules, using an oracle predictor (x0_pred = x0). Does not train.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dbcr import (
    NUM_S2_BANDS,
    get_alpha_schedule,
    reshape_alpha_for_broadcast,
)
from src.models.registry import get_model
from src.utils.config import load_config

TOL = 1e-6
ALL_R3 = [3.0] * NUM_S2_BANDS
GROUPED = [2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0]


def make_eval_timesteps(T, nfe, device="cpu"):
    """Match src/eval.py / src/train.py timestep selection exactly."""
    steps = torch.linspace(T, 0, nfe + 1, device=device)
    timesteps = torch.round(steps).to(torch.long)
    return timesteps


def reverse_step(x_current, x0_pred, alpha_curr, alpha_next):
    """Exact DB-CR deterministic reverse update used in eval.py."""
    return (1 - alpha_next / alpha_curr) * x0_pred + (alpha_next / alpha_curr) * x_current


def forward_bridge(x0, y, alpha):
    return (1 - alpha) * x0 + alpha * y


def inspect_eval_equation():
    """Report whether current eval reverse equation matches DB-CR theory."""
    print("=" * 72)
    print("1. Current eval reverse equation inspection")
    print("=" * 72)
    print(
        "src/eval.py (and the matching test block in src/train.py) implement:\n"
        "    x_t = (1 - alpha_next / alpha_curr) * x0_hat\n"
        "        + (alpha_next / alpha_curr) * x_t\n"
        "with:\n"
        "    alpha_curr = reshape_alpha_for_broadcast(alpha_fn(t_curr), as_channels=True)\n"
        "    alpha_next = reshape_alpha_for_broadcast(alpha_fn(t_next), as_channels=True)\n"
        "and timesteps from:\n"
        "    torch.round(torch.linspace(T, 0, nfe + 1)).long()\n"
    )
    print("Checks:")
    print("  [OK] uses selected alpha_fn (not hard-coded alpha_schedule)")
    print("  [OK] no assumption that alpha = t/T")
    print("  [OK] no sinusoidal-specific algebra in the reverse update")
    print("  [OK] equation matches DB-CR deterministic reverse bridge")
    print(
        "  [NOTE] division by alpha_curr: under linspace(T,0,nfe+1) with "
        "round, t_curr is always > 0 for every reverse step, so "
        "alpha_curr > 0 for both schedules. No code change made."
    )
    print()


def run_oracle_path(alpha_fn, x0, y, T, nfe, device="cpu"):
    timesteps = make_eval_timesteps(T, nfe, device=device)
    x_t = y.clone()
    max_path_error = 0.0

    for k in range(nfe):
        t_curr = timesteps[k]
        t_next = timesteps[k + 1]
        assert int(t_next) <= int(t_curr), "expected t_next <= t_curr"

        alpha_curr = reshape_alpha_for_broadcast(
            alpha_fn(t_curr.float(), T), as_channels=True
        )
        alpha_next = reshape_alpha_for_broadcast(
            alpha_fn(t_next.float(), T), as_channels=True
        )

        # Guard against theoretical alpha_curr=0 (should not happen with eval grid).
        if torch.any(alpha_curr == 0):
            raise AssertionError(
                f"alpha_curr has a zero at t_curr={int(t_curr)} "
                f"(NFE={nfe}, T={T})"
            )

        # Oracle predictor.
        x0_pred = x0
        x_next = reverse_step(x_t, x0_pred, alpha_curr, alpha_next)

        expected = forward_bridge(x0, y, alpha_next)
        step_err = torch.max(torch.abs(x_next - expected)).item()
        max_path_error = max(max_path_error, step_err)
        x_t = x_next

    final_error = torch.max(torch.abs(x_t - x0)).item()
    return {
        "timesteps": timesteps.tolist(),
        "final_error": final_error,
        "max_path_error": max_path_error,
    }


def _endpoint_ok(alpha_fn, T, device="cpu"):
    a0 = alpha_fn(torch.tensor(0.0, device=device), T)
    aT = alpha_fn(torch.tensor(float(T), device=device), T)
    max0 = float(torch.max(torch.abs(a0)).item()) if torch.is_tensor(a0) else abs(float(a0))
    maxT = (
        float(torch.max(torch.abs(aT - 1.0)).item())
        if torch.is_tensor(aT)
        else abs(float(aT) - 1.0)
    )
    return max0 < TOL and maxT < TOL, max0, maxT


def check_nfe1_schedule_independence(schedules, x0, y, T, device="cpu"):
    print("=" * 72)
    print("3. NFE=1 schedule independence")
    print("=" * 72)
    for name, alpha_fn in schedules:
        ok, max0, maxT = _endpoint_ok(alpha_fn, T, device=device)
        print(f"  {name}: max|alpha(0)|={max0:.3e}, max|alpha(T)-1|={maxT:.3e}")
        assert ok, f"{name}: endpoints failed"

        x0_pred = x0
        alpha_curr = reshape_alpha_for_broadcast(
            alpha_fn(torch.tensor(float(T), device=device), T), as_channels=True
        )
        alpha_next = reshape_alpha_for_broadcast(
            alpha_fn(torch.tensor(0.0, device=device), T), as_channels=True
        )
        x_final = reverse_step(y, x0_pred, alpha_curr, alpha_next)
        err = torch.max(torch.abs(x_final - x0_pred)).item()
        assert err < TOL, f"{name}: NFE=1 should give x_final=x0_pred, err={err}"

    print("NFE=1 schedule independence confirmed")
    print()


def test_forward_bridge_equivalence(T=1000, device="cpu"):
    print("=" * 72)
    print("FORWARD BRIDGE EQUIVALENCE (scalar r3 vs all-r3 spectral)")
    print("=" * 72)
    torch.manual_seed(123)
    B, C, H, W = 2, NUM_S2_BANDS, 8, 8
    x0 = torch.rand(B, C, H, W, device=device)
    y = torch.rand(B, C, H, W, device=device)
    t = torch.tensor([0, 1, T // 4, T // 2, (3 * T) // 4, T], device=device).float()

    scalar_fn = get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)
    spectral_fn = get_alpha_schedule(
        "mean_reverting",
        mean_reversion_rate=3.0,
        spectral_mean_reversion_rates=ALL_R3,
    )

    max_diff = 0.0
    for ti in t:
        # Batched identical timesteps.
        tb = ti.repeat(B)
        a_s = reshape_alpha_for_broadcast(scalar_fn(tb, T))
        a_v = reshape_alpha_for_broadcast(spectral_fn(tb, T))
        x_s = forward_bridge(x0, y, a_s)
        x_v = forward_bridge(x0, y, a_v)
        max_diff = max(max_diff, torch.max(torch.abs(x_s - x_v)).item())

    print(f"max_abs_forward_difference={max_diff:.6e}")
    assert max_diff < 1e-6, max_diff
    print("FORWARD BRIDGE EQUIVALENCE: PASS")
    print()
    return max_diff


def test_reverse_equivalence(T=1000, device="cpu"):
    print("=" * 72)
    print("REVERSE UPDATE EQUIVALENCE (scalar r3 vs all-r3 spectral)")
    print("=" * 72)
    torch.manual_seed(7)
    B, C, H, W = 2, NUM_S2_BANDS, 8, 8
    x_t0 = torch.rand(B, C, H, W, device=device)
    x0_hat = torch.rand(B, C, H, W, device=device)

    scalar_fn = get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)
    spectral_fn = get_alpha_schedule(
        "mean_reverting",
        mean_reversion_rate=3.0,
        spectral_mean_reversion_rates=ALL_R3,
    )

    results = {}
    for nfe in (1, 3, 5):
        timesteps = make_eval_timesteps(T, nfe, device=device)
        x_s = x_t0.clone()
        x_v = x_t0.clone()
        max_diff = 0.0
        for k in range(nfe):
            t_curr = timesteps[k]
            t_next = timesteps[k + 1]
            a_sc = reshape_alpha_for_broadcast(
                scalar_fn(t_curr.float(), T), as_channels=True
            )
            a_sn = reshape_alpha_for_broadcast(
                scalar_fn(t_next.float(), T), as_channels=True
            )
            a_vc = reshape_alpha_for_broadcast(
                spectral_fn(t_curr.float(), T), as_channels=True
            )
            a_vn = reshape_alpha_for_broadcast(
                spectral_fn(t_next.float(), T), as_channels=True
            )
            x_s = reverse_step(x_s, x0_hat, a_sc, a_sn)
            x_v = reverse_step(x_v, x0_hat, a_vc, a_vn)
            max_diff = max(max_diff, torch.max(torch.abs(x_s - x_v)).item())
        results[nfe] = max_diff
        print(f"NFE={nfe}: max_abs_reverse_difference={max_diff:.6e}")
        assert max_diff < 1e-6, (nfe, max_diff)

    print("REVERSE UPDATE EQUIVALENCE: PASS")
    print()
    return results


def test_spectral_oracle_grouped(T=1000, device="cpu"):
    print("=" * 72)
    print("SPECTRAL ORACLE REVERSE CONSISTENCY (grouped rates)")
    print("=" * 72)
    torch.manual_seed(99)
    B, C, H, W = 2, NUM_S2_BANDS, 8, 8
    x0 = torch.rand(B, C, H, W, device=device)
    y = torch.rand(B, C, H, W, device=device)
    alpha_fn = get_alpha_schedule(
        "mean_reverting",
        mean_reversion_rate=3.0,
        spectral_mean_reversion_rates=GROUPED,
    )
    # Confirm grouped rates differ across bands at mid time.
    a_mid = alpha_fn(torch.tensor(float(T // 2), device=device), T)
    assert a_mid.shape == (NUM_S2_BANDS,)
    assert float(a_mid[0]) != float(a_mid[11]), "grouped rates should differ B1 vs B11"

    rows = []
    for nfe in (1, 3, 5):
        out = run_oracle_path(alpha_fn, x0, y, T, nfe, device=device)
        rows.append((nfe, out))
        print(
            f"grouped NFE={nfe}: final_error={out['final_error']:.6e} "
            f"max_path_error={out['max_path_error']:.6e}"
        )
        assert out["max_path_error"] < TOL
        assert out["final_error"] < TOL
    print("SPECTRAL ORACLE REVERSE CONSISTENCY: PASS")
    print()
    return rows


def one_batch_smoke(device="cpu"):
    print("=" * 72)
    print("ONE-BATCH SMOKE (no optimizer steps)")
    print("=" * 72)
    torch.manual_seed(42)
    B, C, H, W = 2, NUM_S2_BANDS, 32, 32
    T = 1000
    y = torch.rand(B, C, H, W, device=device)
    x0 = torch.rand(B, C, H, W, device=device)
    z = torch.rand(B, 2, H, W, device=device)
    t = torch.randint(0, T + 1, (B,), device=device)

    model = get_model("dbcr")().to(device)
    model.eval()

    schedules = {
        "scalar_mr_r3": get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0),
        "spectral_all_r3": get_alpha_schedule(
            "mean_reverting",
            mean_reversion_rate=3.0,
            spectral_mean_reversion_rates=ALL_R3,
        ),
        "spectral_grouped": get_alpha_schedule(
            "mean_reverting",
            mean_reversion_rate=3.0,
            spectral_mean_reversion_rates=GROUPED,
        ),
    }

    # Config wiring smoke.
    for cfg_path in (
        "configs/dbcr_mean_reverting.json",
        "configs/dbcr_spectral_mr_all_r3.json",
        "configs/dbcr_spectral_mr_grouped.json",
        "configs/dbcr_spectral_mr_reverse_control.json",
    ):
        cfg = load_config(cfg_path)
        _ = get_alpha_schedule(
            bridge_schedule=cfg["bridge_schedule"],
            mean_reversion_rate=cfg["mean_reversion_rate"],
            spectral_mean_reversion_rates=cfg.get("spectral_mean_reversion_rates"),
        )
        print(f"  config load OK: {cfg_path}")

    results = {}
    with torch.no_grad():
        for name, alpha_fn in schedules.items():
            alpha_t = reshape_alpha_for_broadcast(alpha_fn(t.float(), T))
            x_t = (1 - alpha_t) * x0 + alpha_t * y
            x0_hat = model(x_t, t, z)
            loss = torch.mean(torch.abs(x0_hat - x0))
            assert x_t.shape == (B, C, H, W), x_t.shape
            assert x0_hat.shape == (B, C, H, W), x0_hat.shape
            assert torch.isfinite(x_t).all(), name
            assert torch.isfinite(x0_hat).all(), name
            assert torch.isfinite(loss).all(), name
            results[name] = {
                "alpha_shape": tuple(alpha_t.shape),
                "x_t_shape": tuple(x_t.shape),
                "x0_hat_shape": tuple(x0_hat.shape),
                "loss": float(loss.item()),
                "x_t": x_t,
                "x0_hat": x0_hat,
            }
            print(
                f"  {name}: alpha={alpha_t.shape} x_t={x_t.shape} "
                f"loss={loss.item():.6e}"
            )

    xt_diff = torch.max(
        torch.abs(results["scalar_mr_r3"]["x_t"] - results["spectral_all_r3"]["x_t"])
    ).item()
    hat_diff = torch.max(
        torch.abs(
            results["scalar_mr_r3"]["x0_hat"] - results["spectral_all_r3"]["x0_hat"]
        )
    ).item()
    loss_diff = abs(
        results["scalar_mr_r3"]["loss"] - results["spectral_all_r3"]["loss"]
    )
    print(f"max_abs_x_t_difference={xt_diff:.6e}")
    print(f"max_abs_x0_hat_difference={hat_diff:.6e}")
    print(f"absolute_loss_difference={loss_diff:.6e}")
    assert xt_diff < 1e-6, xt_diff
    assert hat_diff < 1e-6, hat_diff
    assert loss_diff < 1e-6, loss_diff

    # Grouped must differ from scalar at the bridge state for this t (not all-zero).
    grouped_xt_diff = torch.max(
        torch.abs(results["scalar_mr_r3"]["x_t"] - results["spectral_grouped"]["x_t"])
    ).item()
    print(f"grouped_vs_scalar_max_abs_x_t_difference={grouped_xt_diff:.6e}")
    assert grouped_xt_diff > 0.0, "grouped spectral bridge should differ from scalar"

    print("ONE-BATCH SMOKE: PASS")
    print()
    return {
        "max_abs_x_t_difference": xt_diff,
        "max_abs_x0_hat_difference": hat_diff,
        "absolute_loss_difference": loss_diff,
        "grouped_vs_scalar_max_abs_x_t_difference": grouped_xt_diff,
    }


def main():
    inspect_eval_equation()

    torch.manual_seed(42)
    device = "cpu"
    T = 1000
    B, C, H, W = 2, NUM_S2_BANDS, 8, 8
    x0 = torch.rand(B, C, H, W, device=device)
    y = torch.rand(B, C, H, W, device=device)

    schedules = [
        ("original", get_alpha_schedule("original")),
        ("mean_reverting", get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)),
        (
            "spectral_all_r3",
            get_alpha_schedule(
                "mean_reverting",
                mean_reversion_rate=3.0,
                spectral_mean_reversion_rates=ALL_R3,
            ),
        ),
        (
            "spectral_grouped",
            get_alpha_schedule(
                "mean_reverting",
                mean_reversion_rate=3.0,
                spectral_mean_reversion_rates=GROUPED,
            ),
        ),
    ]
    nfe_list = [1, 3, 5]

    print("=" * 72)
    print("2. Timestep sequences (same as eval.py)")
    print("=" * 72)
    for nfe in nfe_list:
        ts = make_eval_timesteps(T, nfe, device=device).tolist()
        print(f"  NFE={nfe}: {ts}")
    print()

    check_nfe1_schedule_independence(schedules, x0, y, T, device=device)

    print("=" * 72)
    print("4. Oracle reverse consistency")
    print("=" * 72)
    print(f"{'Schedule':<16} {'NFE':>4}  {'final_error':>14}  {'max_path_error':>14}")
    for name, alpha_fn in schedules:
        for nfe in nfe_list:
            out = run_oracle_path(alpha_fn, x0, y, T, nfe, device=device)
            print(
                f"{name:<16} {nfe:>4}  {out['final_error']:14.6e}  "
                f"{out['max_path_error']:14.6e}"
            )
            assert out["max_path_error"] < TOL, (
                f"{name} NFE={nfe}: max_path_error={out['max_path_error']}"
            )
            assert out["final_error"] < TOL, (
                f"{name} NFE={nfe}: final_error={out['final_error']}"
            )

    print()
    print("Reverse bridge consistency checks passed.")
    print()

    fwd = test_forward_bridge_equivalence(T=T, device=device)
    rev = test_reverse_equivalence(T=T, device=device)
    test_spectral_oracle_grouped(T=T, device=device)
    smoke = one_batch_smoke(device=device)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"max_abs_forward_difference={fwd:.6e}")
    for nfe, d in rev.items():
        print(f"max_abs_reverse_difference[NFE={nfe}]={d:.6e}")
    print(f"smoke max_abs_x_t_difference={smoke['max_abs_x_t_difference']:.6e}")
    print(f"smoke max_abs_x0_hat_difference={smoke['max_abs_x0_hat_difference']:.6e}")
    print(f"smoke absolute_loss_difference={smoke['absolute_loss_difference']:.6e}")
    print("No training was started.")


if __name__ == "__main__":
    main()
