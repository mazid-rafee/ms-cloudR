#!/usr/bin/env python3
"""Implementation validation for DBCR_MR_r3_SARAlign (12 checks). No training run."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


DEFAULT_TEACHER = (
    "outputs/DBCR_MR_r3_seed42_epochs50_20260812/checkpoints/best.pt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/DBCR_MR_r3_SARAlign/smoke_validation.json",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from src.models.dbcr import DBCRNet, get_alpha_schedule
    from src.utils.sar_align import (
        assert_teacher_frozen,
        extract_detached_optical_anchor,
        grad_norm,
        load_frozen_teacher,
        optimizer_contains_params,
        parameter_group_tensors,
        spatial_cosine_align_loss,
        teacher_requires_grad_any,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"checks": {}, "device": str(device), "teacher": args.teacher}

    # --- 1–4: frozen teacher load ---
    teacher = load_frozen_teacher(args.teacher, device=device)
    assert_teacher_frozen(teacher)
    results["checks"]["1_teacher_checkpoint_loads"] = True
    results["checks"]["2_teacher_requires_grad_false"] = not teacher_requires_grad_any(
        teacher
    )
    results["checks"]["3_teacher_eval_mode"] = not teacher.training

    student = DBCRNet().to(device)
    opt = torch.optim.Adam(student.parameters(), lr=5e-5)
    results["checks"]["4_teacher_absent_from_optimizer"] = not optimizer_contains_params(
        opt, teacher.parameters()
    )

    # Synthetic batch (avoids dataset dependency for core graph checks).
    b, h, w = args.batch_size, 256, 256
    y = torch.rand(b, 13, h, w, device=device)
    z = torch.rand(b, 2, h, w, device=device)
    x0 = torch.rand(b, 13, h, w, device=device)
    t = torch.randint(0, 1001, (b,), device=device)
    alpha_fn = get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)
    alpha_t = alpha_fn(t.float(), 1000).view(-1, 1, 1, 1)
    x_t = (1 - alpha_t) * x0 + alpha_t * y

    # --- 5–7: Fc detach, shapes, finite L_align ---
    fc = extract_detached_optical_anchor(teacher, x0, anchor_t=0.0)
    results["checks"]["5_fc_detached"] = bool(
        (not fc.requires_grad) and (fc.grad_fn is None)
    )

    student.train()
    x0_hat, aux = student(x_t, t, z, return_aux=True)
    fs = aux["sar_stage3"]
    shape_ok = (
        tuple(fs.shape) == (b, 176, 32, 32)
        and tuple(fc.shape) == (b, 176, 32, 32)
        and tuple(fs.shape) == tuple(fc.shape)
    )
    results["checks"]["6_fs_fc_shapes"] = {
        "ok": shape_ok,
        "fs": list(fs.shape),
        "fc": list(fc.shape),
        "x0_hat": list(x0_hat.shape),
    }

    l_recon = torch.mean(torch.abs(x0_hat - x0))
    l_align = spatial_cosine_align_loss(fs, fc)
    results["checks"]["7_l_align_finite"] = bool(
        torch.isfinite(l_align).item() and torch.isfinite(l_recon).item()
    )
    results["sample_losses"] = {
        "l_recon": float(l_recon.item()),
        "l_align": float(l_align.item()),
    }

    # --- 8–9: gradient flow ---
    # Align-only backward: should hit SAR + shared downs, not teacher / not optical-only.
    student.zero_grad(set_to_none=True)
    for p in teacher.parameters():
        if p.grad is not None:
            p.grad = None
    l_align.backward(retain_graph=True)
    groups = parameter_group_tensors(student)
    g_sar = grad_norm(groups["sar_only"])
    g_downs = grad_norm(groups["shared_downs"])
    g_opt = grad_norm(groups["optical_student"])
    teacher_grad = grad_norm(list(teacher.parameters()))
    results["checks"]["8_backward_grads_student_sar_path"] = {
        "ok": g_sar > 0.0 and g_downs > 0.0,
        "grad_norm_sar_only": g_sar,
        "grad_norm_shared_downs": g_downs,
        "grad_norm_optical_student_align_only": g_opt,
    }
    results["checks"]["9_teacher_grad_count_zero"] = {
        "ok": teacher_grad == 0.0,
        "grad_norm_teacher": teacher_grad,
        "teacher_requires_grad_any": teacher_requires_grad_any(teacher),
    }
    # Optical-only params should receive no L_align grads (shared downs excluded).
    results["checks"]["9b_optical_only_zero_from_align"] = {
        "ok": g_opt == 0.0,
        "grad_norm_optical_student_align_only": g_opt,
    }

    # --- 10: x0_hat unchanged by enabling aux / computing L_align (loss-only) ---
    student.zero_grad(set_to_none=True)
    with torch.no_grad():
        # Re-init comparison with fixed weights: forward with/without return_aux.
        pass
    student.eval()
    with torch.no_grad():
        pred_plain = student(x_t, t, z)
        pred_aux, aux2 = student(x_t, t, z, return_aux=True)
        # Computing alignment on detached features cannot change pred.
        _ = spatial_cosine_align_loss(aux2["sar_stage3"], fc)
    max_diff = float((pred_plain - pred_aux).abs().max().item())
    results["checks"]["10_x0_hat_unchanged_by_aux_api"] = {
        "ok": max_diff == 0.0,
        "max_abs_diff": max_diff,
    }

    # --- 11: inference path does not load/use teacher ---
    # Simulate eval: only student forward(y-as-x_t style).
    student.eval()
    with torch.no_grad():
        _ = student(y, torch.full((b,), 1000.0, device=device), z)
    # No teacher reference required — check eval.py source does not import sar_align.
    eval_path = os.path.join("src", "eval.py")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_src = f.read()
    results["checks"]["11_inference_does_not_use_teacher"] = {
        "ok": ("sar_align" not in eval_src) and ("load_frozen_teacher" not in eval_src),
        "eval_imports_sar_align": "sar_align" in eval_src,
    }

    # --- 12: disabling sar_align reproduces ordinary forward API ---
    student.train()
    out_default = student(x_t, t, z)
    results["checks"]["12_default_forward_unchanged"] = {
        "ok": torch.is_tensor(out_default) and tuple(out_default.shape) == (b, 13, h, w),
        "return_type": type(out_default).__name__,
        "shape": list(out_default.shape),
    }

    # Optional real-data sanity if dataset present.
    try:
        from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
        from src.utils.io_utils import map_seasons

        ds = SEN12MSCRDataset(args.data_dir, seasons=map_seasons("winter"))
        if len(ds) > 0:
            y_r, z_r, x0_r = ds[0]
            y_r = y_r.unsqueeze(0).to(device)
            z_r = z_r.unsqueeze(0).to(device)
            x0_r = x0_r.unsqueeze(0).to(device)
            t_r = torch.tensor([0.0], device=device)
            with torch.no_grad():
                fc_r = extract_detached_optical_anchor(teacher, x0_r, anchor_t=0.0)
                xh, aux_r = student(y_r, t_r, z_r, return_aux=True)
                la = spatial_cosine_align_loss(aux_r["sar_stage3"], fc_r)
            results["real_sample_probe"] = {
                "fc": list(fc_r.shape),
                "fs": list(aux_r["sar_stage3"].shape),
                "l_align": float(la.item()),
                "finite": bool(torch.isfinite(la).item()),
            }
    except Exception as exc:
        results["real_sample_probe"] = {"skipped": True, "reason": str(exc)}

    all_ok = True
    for key, val in results["checks"].items():
        if isinstance(val, dict):
            ok = bool(val.get("ok", False))
        else:
            ok = bool(val)
        if not ok:
            all_ok = False
    results["all_passed"] = all_ok

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")
    print("ALL PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
