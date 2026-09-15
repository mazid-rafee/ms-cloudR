#!/usr/bin/env python3
"""Short loss-scale probe for SARAlign lambda (no full epochs).

Runs a fixed number of training steps for each candidate lambda and reports
average L_recon, L_align, weighted align, and ratio.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


DEFAULT_TEACHER = (
    "outputs/DBCR_MR_r3_seed42_epochs50_20260812/checkpoints/best.pt"
)
DEFAULT_LAMBDAS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--mean_reversion_rate", type=float, default=3.0)
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--anchor_t", type=float, default=0.0)
    parser.add_argument(
        "--lambdas",
        type=str,
        default="",
        help="Comma-separated lambda values (default: full candidate grid).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/DBCR_MR_r3_SARAlign/lambda_probe.json",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.lambdas.strip():
        LAMBDAS = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    else:
        LAMBDAS = list(DEFAULT_LAMBDAS)

    from torch.utils.data import DataLoader, Subset
    from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
    from src.models.dbcr import DBCRNet, get_alpha_schedule
    from src.utils.io_utils import map_seasons
    from src.utils.sar_align import (
        assert_teacher_frozen,
        extract_detached_optical_anchor,
        grad_norm,
        load_frozen_teacher,
        parameter_group_tensors,
        spatial_cosine_align_loss,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    seasons = map_seasons("winter,summer,fall,spring")
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    if len(dataset) == 0:
        raise RuntimeError("No SEN12MS-CR samples found for lambda probe")

    # Match controlled split seed, then take a small train prefix for the probe.
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, _, _ = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Cap probe data so DataLoader is light; steps reuse batches via shuffle.
    probe_n = min(len(train_ds), max(args.batch_size * args.steps, 512))
    probe_ds = Subset(train_ds, list(range(probe_n)))
    loader = DataLoader(
        probe_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    alpha_fn = get_alpha_schedule(
        bridge_schedule="mean_reverting",
        mean_reversion_rate=args.mean_reversion_rate,
    )
    teacher = load_frozen_teacher(args.teacher, device=device)
    assert_teacher_frozen(teacher)

    rows = []
    target_lo, target_hi = 0.05, 0.20

    for lam in LAMBDAS:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        student = DBCRNet().to(device)
        opt = torch.optim.Adam(student.parameters(), lr=args.lr)
        groups = parameter_group_tensors(student)
        student.train()
        teacher.eval()

        recon_vals = []
        align_vals = []
        walign_vals = []
        ratio_vals = []
        finite_ok = True
        grad_ok = True
        last_grads = {}

        step = 0
        data_iter = iter(loader)
        while step < args.steps:
            try:
                y, z, x0 = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                y, z, x0 = next(data_iter)

            y = y.to(device)
            z = z.to(device)
            x0 = x0.to(device)
            t = torch.randint(
                0, args.diffusion_steps + 1, (x0.size(0),), device=device
            )
            alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
            x_t = (1 - alpha_t) * x0 + alpha_t * y

            x0_hat, aux = student(x_t, t, z, return_aux=True)
            fs = aux["sar_stage3"]
            fc = extract_detached_optical_anchor(teacher, x0, anchor_t=args.anchor_t)
            l_recon = torch.mean(torch.abs(x0_hat - x0))
            l_align = spatial_cosine_align_loss(fs, fc)
            l_walign = lam * l_align
            loss = l_recon + l_walign

            if not (
                torch.isfinite(l_recon)
                and torch.isfinite(l_align)
                and torch.isfinite(loss)
            ):
                finite_ok = False

            opt.zero_grad()
            loss.backward()
            g_sar = grad_norm(groups["sar_only"])
            g_downs = grad_norm(groups["shared_downs"])
            g_opt = grad_norm(groups["optical_student"])
            last_grads = {
                "grad_norm_sar_only": g_sar,
                "grad_norm_shared_downs": g_downs,
                "grad_norm_optical_student": g_opt,
            }
            if not (g_sar == g_sar and g_downs == g_downs and g_opt == g_opt):
                grad_ok = False
            # NaN check
            for g in (g_sar, g_downs, g_opt):
                if g != g:  # NaN
                    grad_ok = False
            opt.step()

            recon_vals.append(float(l_recon.item()))
            align_vals.append(float(l_align.item()))
            walign_vals.append(float(l_walign.item()))
            ratio_vals.append(float(l_walign.item()) / max(float(l_recon.item()), 1e-12))
            step += 1
            if step % 20 == 0 or step == args.steps:
                print(
                    f"lambda={lam:g} step={step}/{args.steps} "
                    f"recon={recon_vals[-1]:.5f} align={align_vals[-1]:.5f} "
                    f"ratio={ratio_vals[-1]:.4f}",
                    flush=True,
                )

        mean_recon = sum(recon_vals) / len(recon_vals)
        mean_align = sum(align_vals) / len(align_vals)
        mean_walign = sum(walign_vals) / len(walign_vals)
        mean_ratio = sum(ratio_vals) / len(ratio_vals)
        in_target = target_lo <= mean_ratio <= target_hi
        row = {
            "lambda": lam,
            "steps": args.steps,
            "mean_L_recon": mean_recon,
            "mean_L_align": mean_align,
            "mean_lambda_L_align": mean_walign,
            "mean_weighted_align_to_recon_ratio": mean_ratio,
            "ratio_in_5_20_percent": in_target,
            "all_losses_finite": finite_ok,
            "grads_finite": grad_ok,
            "last_grad_norms": last_grads,
            "recon_first": recon_vals[0],
            "recon_last": recon_vals[-1],
        }
        rows.append(row)
        print(
            f"=== lambda={lam:g} mean_recon={mean_recon:.6f} "
            f"mean_align={mean_align:.6f} mean_walign={mean_walign:.6f} "
            f"ratio={mean_ratio:.4f} target={in_target} ===",
            flush=True,
        )

    # Recommend lambda closest to band center 0.125 among those in band;
    # else closest to 0.125 overall.
    def score(r):
        ratio = r["mean_weighted_align_to_recon_ratio"]
        in_band = 1 if target_lo <= ratio <= target_hi else 0
        return (-in_band, abs(ratio - 0.125))

    ranked = sorted(rows, key=score)
    recommended = ranked[0]["lambda"]

    payload = {
        "protocol": {
            "steps_per_lambda": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lr": args.lr,
            "bridge_schedule": "mean_reverting",
            "mean_reversion_rate": args.mean_reversion_rate,
            "anchor_t": args.anchor_t,
            "teacher": args.teacher,
            "target_ratio_band": [target_lo, target_hi],
            "note": (
                "Heuristic loss-ratio target only; not a performance claim. "
                "Student trained from scratch for each lambda probe."
            ),
        },
        "results": rows,
        "recommended_lambda_for_3epoch_pilot": recommended,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Recommended lambda for 3-epoch pilot: {recommended}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
