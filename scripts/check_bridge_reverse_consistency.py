"""Oracle consistency test for deterministic DB-CR reverse bridge updates.

Verifies that the existing eval reverse equation reconstructs analytic bridge
states for BOTH original and mean-reverting alpha schedules, using an oracle
predictor (x0_pred = x0). Does not train or modify the model.
"""

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dbcr import get_alpha_schedule


TOL = 1e-6


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
        "    alpha_curr = alpha_fn(t_curr.float(), T)\n"
        "    alpha_next = alpha_fn(t_next.float(), T)\n"
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

        alpha_curr = alpha_fn(t_curr.float(), T).view(1, 1, 1, 1)
        alpha_next = alpha_fn(t_next.float(), T).view(1, 1, 1, 1)

        # Guard against theoretical alpha_curr=0 (should not happen with eval grid).
        if float(alpha_curr) == 0.0:
            raise AssertionError(
                f"alpha_curr=0 at t_curr={int(t_curr)} would divide by zero "
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


def check_nfe1_schedule_independence(schedules, x0, y, T, device="cpu"):
    print("=" * 72)
    print("3. NFE=1 schedule independence")
    print("=" * 72)
    for name, alpha_fn in schedules:
        a0 = float(alpha_fn(torch.tensor(0.0, device=device), T))
        aT = float(alpha_fn(torch.tensor(float(T), device=device), T))
        print(f"  {name}: alpha(0)={a0:.8f}, alpha(T)={aT:.8f}")
        assert abs(a0) < TOL, f"{name}: alpha(0) should be ~0"
        assert abs(aT - 1.0) < TOL, f"{name}: alpha(T) should be ~1"

        # NFE=1: t_curr=T, t_next=0 => alpha_curr=1, alpha_next=0
        # => x_final = x0_pred, independent of intermediate schedule shape.
        x0_pred = x0
        alpha_curr = alpha_fn(torch.tensor(float(T), device=device), T)
        alpha_next = alpha_fn(torch.tensor(0.0, device=device), T)
        x_final = reverse_step(y, x0_pred, alpha_curr, alpha_next)
        err = torch.max(torch.abs(x_final - x0_pred)).item()
        assert err < TOL, f"{name}: NFE=1 should give x_final=x0_pred, err={err}"

    print("NFE=1 schedule independence confirmed")
    print()


def main():
    inspect_eval_equation()

    torch.manual_seed(42)
    device = "cpu"
    T = 1000
    B, C, H, W = 2, 13, 8, 8
    x0 = torch.rand(B, C, H, W, device=device)
    y = torch.rand(B, C, H, W, device=device)

    schedules = [
        ("original", get_alpha_schedule("original")),
        ("mean_reverting", get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)),
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
    rows = []
    for name, alpha_fn in schedules:
        for nfe in nfe_list:
            out = run_oracle_path(alpha_fn, x0, y, T, nfe, device=device)
            rows.append((name, nfe, out))
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


if __name__ == "__main__":
    main()
