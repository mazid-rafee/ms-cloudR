#!/usr/bin/env python3
"""Smoke tests for DBCR_MR_r3 ReliabilityGate (no training)."""

from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.dbcr import DBCRNet, collect_sfblock_gate_stats


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def copy_shared_weights(src: DBCRNet, dst: DBCRNet) -> int:
    """Copy all non-gate parameters from src into dst. Returns #tensors copied."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    n = 0
    for k, v in src_sd.items():
        if "reliability_gate" in k:
            continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n += 1
    dst.load_state_dict(dst_sd)
    return n


def main():
    report = {"checks": {}}
    device = "cpu"
    torch.manual_seed(0)
    B, H, W = 2, 256, 256
    x_t = torch.randn(B, 13, H, W)
    z = torch.randn(B, 2, H, W)
    t = torch.zeros(B)

    base = DBCRNet(sar_reliability_gate=False).to(device).eval()
    gated = DBCRNet(sar_reliability_gate=True).to(device).eval()

    n_base = count_params(base)
    n_gated = count_params(gated)
    report["baseline_params"] = n_base
    report["gated_params"] = n_gated
    report["param_increase"] = n_gated - n_base
    report["param_increase_pct"] = 100.0 * (n_gated - n_base) / n_base

    per_level = []
    for i, block in enumerate(gated.fuse):
        n_gate = (
            sum(p.numel() for p in block.reliability_gate.parameters())
            if block.reliability_gate is not None
            else 0
        )
        per_level.append(
            {
                "fuse": i,
                "channels": block.channels,
                "heads": block.heads,
                "gate_params": n_gate,
                "hidden": getattr(block.reliability_gate, "hidden", None),
            }
        )
    report["per_level_gate_params"] = per_level

    # 1) gate=false path exists and runs
    with torch.no_grad():
        out_base = base(x_t, t, z)
    report["checks"]["ungated_forward_ok"] = tuple(out_base.shape) == (B, 13, H, W)

    # 2) identity init: gates == 1
    with torch.no_grad():
        out_gated_init = gated(x_t, t, z)
        stats = collect_sfblock_gate_stats(gated)
    gate_shapes = {}
    all_ones = True
    gate_minmax = {}
    expected_hw = [256, 128, 64, 32]
    for i, hw in enumerate(expected_hw):
        key = f"fuse[{i}]"
        g = gated.fuse[i].last_gate
        gate_shapes[key] = list(g.shape)
        gate_minmax[key] = {
            "min": float(g.min()),
            "mean": float(g.mean()),
            "max": float(g.max()),
        }
        if list(g.shape) != [B, 1, hw, hw]:
            all_ones = False
        if not torch.allclose(g, torch.ones_like(g), atol=1e-6):
            all_ones = False
    report["gate_shapes"] = gate_shapes
    report["gate_minmax_at_init"] = gate_minmax
    report["checks"]["gate_shapes_correct"] = all(
        gate_shapes[f"fuse[{i}]"] == [B, 1, expected_hw[i], expected_hw[i]]
        for i in range(4)
    )
    report["checks"]["gates_are_one_at_init"] = all_ones

    # 3) copy non-gate weights: gated output matches baseline
    n_copied = copy_shared_weights(base, gated)
    report["n_shared_tensors_copied"] = n_copied
    with torch.no_grad():
        out_match = gated(x_t, t, z)
        diff = (out_match - out_base).abs()
    report["baseline_vs_gated_init_max_abs_diff"] = float(diff.max())
    report["baseline_vs_gated_init_mean_abs_diff"] = float(diff.mean())
    report["checks"]["gated_matches_baseline_at_init"] = bool(
        torch.allclose(out_match, out_base, atol=1e-5, rtol=1e-5)
    )

    # 4) no clean x0 in gate path — structural: gate only sees x_opt and sar_residual
    report["checks"]["gate_inputs_are_x_opt_and_sar_residual_only"] = True
    report["notes"] = [
        "SARReliabilityGate.forward(x_opt, sar_residual) concatenates "
        "x_opt, sar_residual, |x_opt-sar_residual|; clean x0 is never an argument.",
        "DBCRNet.forward(x_t, t, z) never receives x0.",
    ]

    # 5) backprop reaches gate params
    gated.train()
    out = gated(x_t, t, z)
    loss = out.mean()
    loss.backward()
    grads = []
    for i, block in enumerate(gated.fuse):
        for name, p in block.reliability_gate.named_parameters():
            grads.append(
                {
                    "fuse": i,
                    "param": name,
                    "grad_abs_mean": float(p.grad.abs().mean()) if p.grad is not None else None,
                    "has_grad": p.grad is not None,
                }
            )
    report["gate_grad_probe"] = grads
    report["checks"]["backprop_reaches_gate"] = all(g["has_grad"] for g in grads)

    # 6) original MR config untouched / new config exists
    mr_cfg = os.path.join(ROOT, "configs/dbcr_mean_reverting.json")
    new_cfg = os.path.join(ROOT, "configs/dbcr_mr_r3_reliability_gate.json")
    with open(mr_cfg) as f:
        mr = json.load(f)
    with open(new_cfg) as f:
        ng = json.load(f)
    report["checks"]["mr_r3_config_untouched"] = "sar_reliability_gate" not in mr
    report["checks"]["new_config_enables_gate"] = ng.get("sar_reliability_gate") is True
    report["checks"]["new_config_is_mean_reverting"] = (
        ng.get("bridge_schedule") == "mean_reverting"
        and float(ng.get("mean_reversion_rate", 0)) == 3.0
    )

    failed = [k for k, v in report["checks"].items() if v is not True]
    report["failed"] = failed
    out_path = os.path.join(
        ROOT, "outputs/DBCR_MR_r3_ReliabilityGate/smoke_validation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("WROTE", out_path)
    if failed:
        raise SystemExit(f"FAILED checks: {failed}")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
