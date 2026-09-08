"""Sanity-check DB-CR alpha schedules (original vs mean-reverting)."""

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dbcr import alpha_schedule, mean_reverting_alpha_schedule


def _format_row(values):
    return " ".join(f"{v:0.6f}" for v in values)


def main():
    T = 1000
    rate = 3.0
    fractions = [0.00, 0.25, 0.50, 0.75, 1.00]
    t = torch.tensor([f * T for f in fractions], dtype=torch.float32)

    original = alpha_schedule(t, T)
    mean_rev = mean_reverting_alpha_schedule(t, T, rate=rate)

    print("Original DB-CR alpha:")
    print("t/T:", " ".join(f"{f:0.2f}" for f in fractions))
    print("alpha:", _format_row(original.tolist()))
    print()
    print(f"Mean-reverting alpha (rate={rate}):")
    print("t/T:", " ".join(f"{f:0.2f}" for f in fractions))
    print("alpha:", _format_row(mean_rev.tolist()))
    print()

    assert abs(float(mean_rev[0])) < 1e-6, f"alpha(0) should be ~0, got {mean_rev[0]}"
    assert abs(float(mean_rev[-1]) - 1.0) < 1e-6, f"alpha(T) should be ~1, got {mean_rev[-1]}"
    for i in range(len(mean_rev) - 1):
        assert float(mean_rev[i + 1]) >= float(mean_rev[i]) - 1e-8, (
            f"mean-reverting alpha must be non-decreasing: "
            f"{mean_rev[i]} -> {mean_rev[i + 1]}"
        )

    # Device / dtype sanity: stay on CPU here; CUDA may be unavailable/incompatible
    # in some environments. The schedule still places rate on t.device when used.
    t_cpu = t.cpu()
    out_cpu = mean_reverting_alpha_schedule(t_cpu, T, rate=rate)
    assert out_cpu.device == t_cpu.device

    if torch.cuda.is_available():
        try:
            t_dev = t.cuda()
            out_dev = mean_reverting_alpha_schedule(t_dev, T, rate=rate)
            assert out_dev.device == t_dev.device
            _ = out_dev.cpu()  # force sync; skip if kernel unsupported
        except RuntimeError as exc:
            print(f"CUDA device check skipped ({exc}).")

    try:
        mean_reverting_alpha_schedule(t, T, rate=0.0)
        raise AssertionError("rate<=0 should raise ValueError")
    except ValueError:
        pass

    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
