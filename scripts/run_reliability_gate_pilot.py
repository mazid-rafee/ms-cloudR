#!/usr/bin/env python3
"""3-epoch ReliabilityGate pilot (from scratch). No interventions. No 50-epoch run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.models.dbcr import (
    DBCRNet,
    collect_sfblock_gate_stats,
    get_alpha_schedule,
    summarize_gate_map,
)
from src.utils.io_utils import map_seasons, save_json
from src.utils.logger import setup_logger, append_metrics_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="4")
    p.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    p.add_argument("--seasons", type=str, default="winter,summer,fall,spring")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--nfe", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mean_reversion_rate", type=float, default=3.0)
    p.add_argument(
        "--run_name",
        type=str,
        default="DBCR_MR_r3_ReliabilityGate_seed42_3epoch_pilot",
    )
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--gate_eval_batches", type=int, default=32)
    p.add_argument("--endpoint_eval_batches", type=int, default=32)
    return p.parse_args()


def gate_param_snapshot(model):
    vecs = []
    for block in model.fuse:
        if block.reliability_gate is None:
            continue
        for p in block.reliability_gate.parameters():
            vecs.append(p.detach().float().reshape(-1).cpu())
    return torch.cat(vecs) if vecs else torch.zeros(1)


def gate_grad_norm(model):
    total = 0.0
    n = 0
    for block in model.fuse:
        if block.reliability_gate is None:
            continue
        for p in block.reliability_gate.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if torch.isnan(g).any() or torch.isinf(g).any():
                return float("nan")
            total += float(g.norm().item() ** 2)
            n += 1
    return math.sqrt(total) if n else 0.0


def evaluate_random_t_l1(model, loader, alpha_fn, T, device, max_batches=0):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for step, (y, z, x0) in enumerate(loader, start=1):
            y, z, x0 = y.to(device), z.to(device), x0.to(device)
            t = torch.randint(0, T + 1, (x0.size(0),), device=device)
            alpha_t = alpha_fn(t.float(), T).view(-1, 1, 1, 1)
            x_t = (1 - alpha_t) * x0 + alpha_t * y
            x0_hat = model(x_t, t, z)
            total += torch.mean(torch.abs(x0_hat - x0)).item()
            n += 1
            if max_batches > 0 and step >= max_batches:
                break
    return total / max(1, n)


def evaluate_endpoint_l1(model, loader, alpha_fn, T, nfe, device, max_batches=0):
    """NFE reverse starting from y (same as eval), L1 vs clean x0."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for step, (y, z, x0) in enumerate(loader, start=1):
            y, z, x0 = y.to(device), z.to(device), x0.to(device)
            x_t = y
            steps = torch.linspace(T, 0, nfe + 1, device=device)
            timesteps = torch.round(steps).to(torch.long)
            for k in range(nfe):
                t_curr = timesteps[k]
                t_next = timesteps[k + 1]
                alpha_curr = alpha_fn(t_curr.float(), T).view(1, 1, 1, 1)
                alpha_next = alpha_fn(t_next.float(), T).view(1, 1, 1, 1)
                x0_hat = model(x_t, t_curr.repeat(x_t.size(0)), z)
                x_t = (1 - alpha_next / alpha_curr) * x0_hat + (
                    alpha_next / alpha_curr
                ) * x_t
            total += torch.mean(torch.abs(x0_hat - x0)).item()
            n += 1
            if max_batches > 0 and step >= max_batches:
                break
    return total / max(1, n)


def evaluate_gate_stats(model, loader, alpha_fn, T, device, max_batches=32):
    """Fixed-subset gate stats using random-t forward (eval mode, detached cache)."""
    model.eval()
    acc = {
        f"fuse[{i}]": {
            "sum_mean": 0.0,
            "sum_std": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "sum_frac_lt_0.5": 0.0,
            "sum_frac_lt_0.8": 0.0,
            "sum_frac_gt_1.2": 0.0,
            "sum_frac_gt_1.5": 0.0,
            "sum_mad_from_1": 0.0,
            "n": 0,
            "has_nan": False,
            "has_inf": False,
            "shape": None,
        }
        for i in range(len(model.fuse))
    }
    with torch.no_grad():
        for step, (y, z, x0) in enumerate(loader, start=1):
            y, z, x0 = y.to(device), z.to(device), x0.to(device)
            t = torch.randint(0, T + 1, (x0.size(0),), device=device)
            alpha_t = alpha_fn(t.float(), T).view(-1, 1, 1, 1)
            x_t = (1 - alpha_t) * x0 + alpha_t * y
            _ = model(x_t, t, z)
            assert all(block.last_gate is not None for block in model.fuse)
            # Confirm caches are detached (no autograd graph).
            for block in model.fuse:
                assert not block.last_gate.requires_grad
            stats = collect_sfblock_gate_stats(model)
            for key, rec in stats.items():
                a = acc[key]
                g = model.fuse[int(key[5:-1])].last_gate
                a["sum_mean"] += rec["gate_mean"]
                a["sum_std"] += rec["gate_std"]
                a["min"] = min(a["min"], rec["gate_min"])
                a["max"] = max(a["max"], rec["gate_max"])
                a["sum_frac_lt_0.5"] += rec["frac_lt_0.5"]
                a["sum_frac_lt_0.8"] += rec["frac_lt_0.8"]
                a["sum_frac_gt_1.2"] += rec["frac_gt_1.2"]
                a["sum_frac_gt_1.5"] += rec["frac_gt_1.5"]
                a["sum_mad_from_1"] += float(torch.mean(torch.abs(g - 1.0)).item())
                a["n"] += 1
                a["shape"] = rec["shape"]
                if torch.isnan(g).any():
                    a["has_nan"] = True
                if torch.isinf(g).any():
                    a["has_inf"] = True
            if step >= max_batches:
                break
    out = {}
    for key, a in acc.items():
        n = max(1, a["n"])
        out[key] = {
            "gate_mean": a["sum_mean"] / n,
            "gate_std": a["sum_std"] / n,
            "gate_min": a["min"] if a["min"] != float("inf") else None,
            "gate_max": a["max"] if a["max"] != float("-inf") else None,
            "frac_lt_0.5": a["sum_frac_lt_0.5"] / n,
            "frac_lt_0.8": a["sum_frac_lt_0.8"] / n,
            "frac_gt_1.2": a["sum_frac_gt_1.2"] / n,
            "frac_gt_1.5": a["sum_frac_gt_1.5"] / n,
            "mean_abs_dev_from_1": a["sum_mad_from_1"] / n,
            "has_nan": a["has_nan"],
            "has_inf": a["has_inf"],
            "n_batches": a["n"],
            "shape": a["shape"],
        }
    return out


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.output_dir, args.run_name)
    log_dir = os.path.join(run_dir, "logs")
    logger = setup_logger(log_dir, "pilot")
    logger.info("device=%s gpu=%s", device, args.gpu)
    logger.info(
        "ReliabilityGate 3-epoch pilot seed=%s MR_r3 rate=%s from_scratch",
        args.seed,
        args.mean_reversion_rate,
    )

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    seasons = map_seasons(args.seasons)
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info("split train/val/test=%d/%d/%d", train_size, val_size, test_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    # Fixed gate-eval subset: first N batches of val loader (shuffle=False).
    gate_loader = val_loader

    alpha_fn = get_alpha_schedule("mean_reverting", args.mean_reversion_rate)
    model = DBCRNet(sar_reliability_gate=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_gate = sum(
        p.numel()
        for b in model.fuse
        for p in b.reliability_gate.parameters()
    )
    logger.info("trainable_params=%d gate_params=%d", n_params, n_gate)

    init_gate_vec = gate_param_snapshot(model)
    # Confirm training forward clears last_gate
    model.train()
    y0, z0, x00 = next(iter(train_loader))
    y0, z0, x00 = y0.to(device), z0.to(device), x00.to(device)
    t0 = torch.zeros(y0.size(0), device=device, dtype=torch.long)
    _ = model(y0, t0, z0)
    assert all(b.last_gate is None for b in model.fuse), "last_gate must be None in train"
    logger.info("train-mode last_gate cleared: OK")

    save_json(
        os.path.join(run_dir, "config.json"),
        {
            "run_name": args.run_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "bridge_schedule": "mean_reverting",
            "mean_reversion_rate": args.mean_reversion_rate,
            "sar_reliability_gate": True,
            "init": "from_scratch",
            "split_sizes": {"train": train_size, "val": val_size, "test": test_size},
            "trainable_parameters": n_params,
            "gate_parameters": n_gate,
            "gate_eval_batches": args.gate_eval_batches,
            "endpoint_eval_batches": args.endpoint_eval_batches,
        },
    )

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    epoch_rows = []
    T = args.diffusion_steps

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        grad_norms = []
        for step, (y, z, x0) in enumerate(train_loader, start=1):
            y, z, x0 = y.to(device), z.to(device), x0.to(device)
            t = torch.randint(0, T + 1, (x0.size(0),), device=device)
            alpha_t = alpha_fn(t.float(), T).view(-1, 1, 1, 1)
            x_t = (1 - alpha_t) * x0 + alpha_t * y
            x0_hat = model(x_t, t, z)
            loss = torch.mean(torch.abs(x0_hat - x0))
            opt.zero_grad()
            loss.backward()
            grad_norms.append(gate_grad_norm(model))
            opt.step()
            train_loss += loss.item()
            if step % args.log_every == 0 or step == len(train_loader):
                sys.stdout.write(
                    f"\rTrain {epoch} [{step}/{len(train_loader)}] loss={loss.item():.6f}"
                )
                sys.stdout.flush()
        sys.stdout.write("\n")
        train_loss /= max(1, len(train_loader))
        finite_grads = [g for g in grad_norms if g == g]
        gate_grad = {
            "mean": sum(finite_grads) / max(1, len(finite_grads)),
            "max": max(finite_grads) if finite_grads else None,
            "min": min(finite_grads) if finite_grads else None,
            "n_nan": sum(1 for g in grad_norms if g != g),
            "n_steps": len(grad_norms),
        }

        val_random_t = evaluate_random_t_l1(
            model, val_loader, alpha_fn, T, device, max_batches=0
        )
        val_endpoint = evaluate_endpoint_l1(
            model,
            val_loader,
            alpha_fn,
            T,
            args.nfe,
            device,
            max_batches=args.endpoint_eval_batches,
        )
        gate_stats = evaluate_gate_stats(
            model,
            gate_loader,
            alpha_fn,
            T,
            device,
            max_batches=args.gate_eval_batches,
        )
        cur_gate_vec = gate_param_snapshot(model)
        gate_changed = bool(
            not torch.allclose(cur_gate_vec, init_gate_vec, atol=0.0, rtol=0.0)
        )
        gate_param_l2_delta = float(torch.norm(cur_gate_vec - init_gate_vec).item())

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss_random_t": round(val_random_t, 6),
            "val_l1_endpoint_subset": round(val_endpoint, 6),
            "gate_grad_norm_mean": round(gate_grad["mean"], 8),
            "gate_grad_norm_max": round(gate_grad["max"], 8) if gate_grad["max"] is not None else None,
            "gate_params_changed_from_init": gate_changed,
            "gate_param_l2_delta_from_init": round(gate_param_l2_delta, 8),
            "gate_stats": gate_stats,
        }
        epoch_rows.append(row)
        append_metrics_csv(
            metrics_csv,
            {
                "epoch": epoch,
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss_random_t"],
                "val_l1_endpoint_subset": row["val_l1_endpoint_subset"],
                "gate_grad_norm_mean": row["gate_grad_norm_mean"],
                "gate_param_l2_delta_from_init": row["gate_param_l2_delta_from_init"],
            },
            header=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_l1_endpoint_subset",
                "gate_grad_norm_mean",
                "gate_param_l2_delta_from_init",
            ],
        )
        logger.info(
            "Epoch %d train=%.6f val_random_t=%.6f val_endpoint_subset=%.6f "
            "gate_grad_mean=%.6g gate_dL2=%.6g changed=%s",
            epoch,
            train_loss,
            val_random_t,
            val_endpoint,
            gate_grad["mean"],
            gate_param_l2_delta,
            gate_changed,
        )
        for k, st in gate_stats.items():
            logger.info(
                "  %s mean=%.4f std=%.4f min=%.4f max=%.4f mad1=%.4f "
                "lt0.5=%.4f lt0.8=%.4f gt1.2=%.4f gt1.5=%.4f nan=%s inf=%s",
                k,
                st["gate_mean"],
                st["gate_std"],
                st["gate_min"],
                st["gate_max"],
                st["mean_abs_dev_from_1"],
                st["frac_lt_0.5"],
                st["frac_lt_0.8"],
                st["frac_gt_1.2"],
                st["frac_gt_1.5"],
                st["has_nan"],
                st["has_inf"],
            )

    # Historical comparison (first 3 epochs of DBCR_MR_r3 seed42).
    hist_path = os.path.join(
        args.output_dir, "DBCR_MR_r3_seed42_epochs50_20260812", "metrics.csv"
    )
    historical = []
    if os.path.isfile(hist_path):
        with open(hist_path) as f:
            lines = f.read().strip().splitlines()[1:4]
        for line in lines:
            ep, tr, va = line.split(",")
            historical.append(
                {"epoch": int(ep), "train_loss": float(tr), "val_loss": float(va)}
            )

    report = {
        "run_name": args.run_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "bridge_schedule": "mean_reverting",
        "mean_reversion_rate": args.mean_reversion_rate,
        "init": "from_scratch",
        "checkpoint_loaded": False,
        "last_gate_train_policy": "None during training; detach cache only in eval",
        "epochs_detail": epoch_rows,
        "historical_DBCR_MR_r3_seed42_first3": historical,
        "note": (
            "val_loss_random_t matches the DBCR training pipeline validation. "
            "val_l1_endpoint_subset is an extra NFE=1 diagnostic on a fixed-size "
            "val prefix and is not part of the original training loop."
        ),
    }
    save_json(os.path.join(run_dir, "pilot_report.json"), report)
    logger.info("Wrote %s", os.path.join(run_dir, "pilot_report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
