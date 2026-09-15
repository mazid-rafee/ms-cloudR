import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument("--model", type=str, default="dbcr")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--seasons",
        type=str,
        default="winter,summer,fall,spring"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument(
        "--bridge_schedule",
        type=str,
        default="original",
        choices=["original", "mean_reverting"],
        help="DB-CR alpha(t) trajectory: original sinusoidal or mean-reverting.",
    )
    parser.add_argument(
        "--mean_reversion_rate",
        type=float,
        default=3.0,
        help="Rate for mean_reverting bridge schedule (ignored if original).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--num_save_images", type=int, default=4)
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--subset_max", type=int, default=0)
    parser.add_argument(
        "--sar_reliability_gate",
        action="store_true",
        help=(
            "Enable spatial [B,1,H,W] SAR residual gate on SFBlock projected "
            "SAR residual. Default off preserves DBCR_MR_r3."
        ),
    )
    parser.add_argument(
        "--log_gate_stats_every",
        type=int,
        default=5,
        help=(
            "When sar_reliability_gate is on, log fixed-subset gate stats "
            "every N epochs (also logs epoch 1). Set 1 to log every epoch."
        ),
    )
    parser.add_argument(
        "--gate_eval_batches",
        type=int,
        default=32,
        help="Fixed validation prefix size for SAR residual gate stats.",
    )
    args = parser.parse_args()
    defaults = {k: parser.get_default(k) for k in vars(args)}
    return args, defaults


def parse_seasons(value):
    from src.utils.io_utils import map_seasons
    return map_seasons(value)


def progress_bar(prefix, step, total, bar_width=30):
    filled = int(bar_width * step / total)
    bar = "=" * filled + "." * (bar_width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {step}/{total}")
    sys.stdout.flush()


def evaluate_gate_stats(model, loader, alpha_fn, T, device, max_batches=32):
    """Fixed-subset SAR residual gate stats (eval mode; detached last_gate)."""
    from src.models.dbcr import collect_sfblock_gate_stats

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
    import torch

    with torch.no_grad():
        for step, (y, z, x0) in enumerate(loader, start=1):
            y, z, x0 = y.to(device), z.to(device), x0.to(device)
            t = torch.randint(0, T + 1, (x0.size(0),), device=device)
            alpha_t = alpha_fn(t.float(), T).view(-1, 1, 1, 1)
            x_t = (1 - alpha_t) * x0 + alpha_t * y
            _ = model(x_t, t, z)
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
                a["sum_mad_from_1"] += float(
                    rec.get(
                        "mean_abs_dev_from_1",
                        float(torch.mean(torch.abs(g - 1.0)).item()),
                    )
                )
                a["n"] += 1
                a["shape"] = rec["shape"]
                a["has_nan"] = a["has_nan"] or bool(rec.get("has_nan", False))
                a["has_inf"] = a["has_inf"] or bool(rec.get("has_inf", False))
            if max_batches > 0 and step >= max_batches:
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
    args, defaults = parse_args()
    if args.config:
        from src.utils.config import load_config, apply_config
        args = apply_config(args, load_config(args.config), defaults=defaults)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.data import DataLoader, random_split, Subset
    from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
    from src.models.dbcr import get_alpha_schedule
    from src.models.registry import get_model
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    from src.utils.io_utils import save_json, utc_timestamp
    from src.utils.logger import setup_logger, append_metrics_csv
    from src.utils.metrics import psnr, ssim, sam_deg
    from src.utils.image_utils import save_npy, save_rgb_png

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or utc_timestamp()
    run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    logger = setup_logger(log_dir, "train")
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("seed=%s", args.seed)
    logger.info(
        "bridge_schedule=%s mean_reversion_rate=%s diffusion_steps=%s",
        args.bridge_schedule,
        args.mean_reversion_rate,
        args.diffusion_steps,
    )
    alpha_fn = get_alpha_schedule(
        bridge_schedule=args.bridge_schedule,
        mean_reversion_rate=args.mean_reversion_rate,
    )

    seasons = parse_seasons(args.seasons)
    logger.info("Building dataset...")
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    logger.info("Dataset size: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("No samples found. Check dataset paths and file naming.")
        return
    if args.subset_frac < 1.0 or args.subset_max > 0:
        max_len = len(dataset)
        frac_len = max(1, int(max_len * args.subset_frac))
        if args.subset_max > 0:
            frac_len = min(frac_len, args.subset_max)
        indices = torch.randperm(max_len)[:frac_len]
        dataset = Subset(dataset, indices.tolist())
        logger.info("Subset size: %d samples", len(dataset))

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    logger.info(
        "Train/Val/Test sizes: %d/%d/%d",
        len(train_ds),
        len(val_ds),
        len(test_ds)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model_cls = get_model(args.model)
    if args.model != "dbcr":
        raise NotImplementedError("Only DB-CR is wired into the training loop right now.")
    model = model_cls(
        sar_reliability_gate=bool(getattr(args, "sar_reliability_gate", False))
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable_parameters=%d", n_params)
    use_sar_residual_gate = bool(getattr(args, "sar_reliability_gate", False))
    log_gate_every = max(1, int(getattr(args, "log_gate_stats_every", 5)))
    gate_eval_batches = max(1, int(getattr(args, "gate_eval_batches", 32)))
    logger.info(
        "sar_reliability_gate=%s (experimental SAR residual gate; "
        "reliability interpretation deferred to post-training interventions)",
        use_sar_residual_gate,
    )
    if use_sar_residual_gate:
        logger.info(
            "gate_stats: every %d epoch(s) (+epoch 1), fixed val prefix=%d batches",
            log_gate_every,
            gate_eval_batches,
        )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_val = None
    gate_stats_csv = os.path.join(run_dir, "gate_stats.csv")
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer=opt, map_location=device)
        start_epoch = payload.get("epoch", 0) + 1
        best_val = payload.get("best_val")
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    save_json(
        os.path.join(run_dir, "config.json"),
        {
            "data_dir": args.data_dir,
            "seasons": seasons,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "lr": args.lr,
            "diffusion_steps": args.diffusion_steps,
            "nfe": args.nfe,
            "bridge_schedule": args.bridge_schedule,
            "mean_reversion_rate": args.mean_reversion_rate,
            "seed": args.seed,
            "run_name": run_name,
            "model": args.model,
            "config": args.config,
            "trainable_parameters": n_params,
            "sar_reliability_gate": use_sar_residual_gate,
            "log_gate_stats_every": log_gate_every if use_sar_residual_gate else None,
            "gate_eval_batches": gate_eval_batches if use_sar_residual_gate else None,
            "split_sizes": {
                "train": train_size,
                "val": val_size,
                "test": test_size
            }
        }
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        logger.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = 0.0
        for step, (y, z, x0) in enumerate(train_loader, start=1):
            y = y.to(device)
            z = z.to(device)
            x0 = x0.to(device)

            t = torch.randint(0, args.diffusion_steps + 1, (x0.size(0),), device=device)
            alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
            x_t = (1 - alpha_t) * x0 + alpha_t * y
            x0_hat = model(x_t, t, z)
            loss = torch.mean(torch.abs(x0_hat - x0))

            if not torch.isfinite(loss):
                logger.error(
                    "Non-finite train loss at epoch %d step %d: %s — stopping.",
                    epoch,
                    step,
                    float(loss.detach().cpu()) if loss.numel() == 1 else "non-scalar",
                )
                return

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            if step % args.log_every == 0 or step == len(train_loader):
                progress_bar(f"Train {epoch}", step, len(train_loader))

        sys.stdout.write("\n")
        train_loss /= max(1, len(train_loader))

        val_loss = None
        if len(val_loader) > 0:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for step, (y, z, x0) in enumerate(val_loader, start=1):
                    y = y.to(device)
                    z = z.to(device)
                    x0 = x0.to(device)

                    t = torch.randint(0, args.diffusion_steps + 1, (x0.size(0),), device=device)
                    alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
                    x_t = (1 - alpha_t) * x0 + alpha_t * y
                    x0_hat = model(x_t, t, z)
                    loss = torch.mean(torch.abs(x0_hat - x0))
                    if not torch.isfinite(loss):
                        logger.error(
                            "Non-finite val loss at epoch %d step %d — stopping.",
                            epoch,
                            step,
                        )
                        return
                    val_total += loss.item()

                    if step % args.log_every == 0 or step == len(val_loader):
                        progress_bar(f"Val {epoch}", step, len(val_loader))
            sys.stdout.write("\n")
            val_loss = val_total / max(1, len(val_loader))

        metrics_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else ""
        }
        append_metrics_csv(metrics_csv, metrics_row, header=["epoch", "train_loss", "val_loss"])
        logger.info("Train loss: %.6f", train_loss)
        if val_loss is not None:
            logger.info("Val loss: %.6f", val_loss)
            # Warning only: substantial deterioration vs best so far (do not stop).
            if best_val is not None and best_val > 0 and val_loss > 1.5 * best_val:
                logger.warning(
                    "VAL DETERIORATION WARNING: val_loss=%.6f is >1.5x best_val=%.6f "
                    "(continuing; not an automatic stop).",
                    val_loss,
                    best_val,
                )

        if (
            use_sar_residual_gate
            and len(val_loader) > 0
            and (epoch == 1 or epoch % log_gate_every == 0)
        ):
            gate_stats = evaluate_gate_stats(
                model,
                val_loader,
                alpha_fn,
                args.diffusion_steps,
                device,
                max_batches=gate_eval_batches,
            )
            means = []
            any_nan_inf = False
            for i in range(4):
                key = f"fuse[{i}]"
                st = gate_stats.get(key)
                if st is None:
                    continue
                means.append(st["gate_mean"])
                any_nan_inf = any_nan_inf or st["has_nan"] or st["has_inf"]
                logger.info(
                    "SAR residual gate %s mean=%.4f std=%.4f min=%.4f max=%.4f "
                    "mad1=%.4f lt0.5=%.4f lt0.8=%.4f gt1.2=%.4f gt1.5=%.4f "
                    "nan=%s inf=%s",
                    key,
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
                append_metrics_csv(
                    gate_stats_csv,
                    {
                        "epoch": epoch,
                        "fuse_level": i,
                        "gate_mean": round(st["gate_mean"], 6),
                        "gate_std": round(st["gate_std"], 6),
                        "gate_min": round(st["gate_min"], 6),
                        "gate_max": round(st["gate_max"], 6),
                        "mean_abs_dev_from_1": round(st["mean_abs_dev_from_1"], 6),
                        "frac_lt_0.5": round(st["frac_lt_0.5"], 6),
                        "frac_lt_0.8": round(st["frac_lt_0.8"], 6),
                        "frac_gt_1.2": round(st["frac_gt_1.2"], 6),
                        "frac_gt_1.5": round(st["frac_gt_1.5"], 6),
                        "has_nan": int(st["has_nan"]),
                        "has_inf": int(st["has_inf"]),
                    },
                    header=[
                        "epoch",
                        "fuse_level",
                        "gate_mean",
                        "gate_std",
                        "gate_min",
                        "gate_max",
                        "mean_abs_dev_from_1",
                        "frac_lt_0.5",
                        "frac_lt_0.8",
                        "frac_gt_1.2",
                        "frac_gt_1.5",
                        "has_nan",
                        "has_inf",
                    ],
                )
            if any_nan_inf:
                logger.error(
                    "NaN/Inf in SAR residual gate maps at epoch %d — stopping.",
                    epoch,
                )
                return
            if len(means) == 4 and all(m < 0.1 for m in means):
                logger.warning(
                    "GATE COLLAPSE WARNING: all four fuse gate means < 0.1 "
                    "at epoch %d (means=%s); continuing without auto-stop.",
                    epoch,
                    [round(m, 4) for m in means],
                )

        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                model,
                opt,
                epoch,
                extra={
                    "best_val": best_val,
                    "bridge_schedule": args.bridge_schedule,
                    "mean_reversion_rate": args.mean_reversion_rate,
                    "sar_reliability_gate": use_sar_residual_gate,
                }
            )

    logger.info("Training complete.")

    if len(test_loader) > 0:
        logger.info("Starting test...")
        model.eval()
        test_l1 = 0.0
        test_psnr = 0.0
        test_ssim = 0.0
        test_sam = 0.0
        test_lpips = 0.0
        saved = 0
        lpips_model = None
        fid_metric = None
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device)
        except Exception as exc:
            logger.warning("LPIPS not available: %s", exc)
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        except Exception as exc:
            logger.warning("FID not available: %s", exc)
        with torch.no_grad():
            for step, (y, z, x0) in enumerate(test_loader, start=1):
                y = y.to(device)
                z = z.to(device)
                x0 = x0.to(device)

                x_t = y
                T = args.diffusion_steps
                steps = torch.linspace(T, 0, args.nfe + 1, device=device)
                timesteps = torch.round(steps).to(torch.long)
                for k in range(args.nfe):
                    t_curr = timesteps[k]
                    t_next = timesteps[k + 1]
                    alpha_curr = alpha_fn(t_curr.float(), T).view(1, 1, 1, 1)
                    alpha_next = alpha_fn(t_next.float(), T).view(1, 1, 1, 1)
                    x0_hat = model(x_t, t_curr.repeat(x_t.size(0)), z)
                    x_t = (1 - alpha_next / alpha_curr) * x0_hat + (alpha_next / alpha_curr) * x_t

                test_l1 += torch.mean(torch.abs(x0_hat - x0)).item()
                test_psnr += psnr(x0_hat, x0).item()
                test_ssim += ssim(x0_hat, x0).item()
                test_sam += sam_deg(x0_hat, x0).item()

                if lpips_model is not None or fid_metric is not None:
                    rgb_pred = x0_hat[:, [3, 2, 1], :, :].clamp(0.0, 1.0)
                    rgb_gt = x0[:, [3, 2, 1], :, :].clamp(0.0, 1.0)
                    if lpips_model is not None:
                        lp_pred = rgb_pred * 2.0 - 1.0
                        lp_gt = rgb_gt * 2.0 - 1.0
                        test_lpips += lpips_model(lp_pred, lp_gt).mean().item()
                    if fid_metric is not None:
                        rgb_pred_u8 = (rgb_pred * 255.0).to(torch.uint8)
                        rgb_gt_u8 = (rgb_gt * 255.0).to(torch.uint8)
                        fid_metric.update(rgb_pred_u8, real=False)
                        fid_metric.update(rgb_gt_u8, real=True)

                if args.save_images and saved < args.num_save_images:
                    out_dir = os.path.join(run_dir, "images")
                    save_npy(os.path.join(out_dir, f"{saved}_cloudy.npy"), y[0])
                    save_npy(os.path.join(out_dir, f"{saved}_clean.npy"), x0[0])
                    save_npy(os.path.join(out_dir, f"{saved}_pred.npy"), x0_hat[0])
                    save_rgb_png(os.path.join(out_dir, f"{saved}_cloudy.png"), y[0])
                    save_rgb_png(os.path.join(out_dir, f"{saved}_clean.png"), x0[0])
                    save_rgb_png(os.path.join(out_dir, f"{saved}_pred.png"), x0_hat[0])
                    saved += 1

                if step % args.log_every == 0 or step == len(test_loader):
                    progress_bar("Test", step, len(test_loader))
        sys.stdout.write("\n")
        test_l1 /= max(1, len(test_loader))
        test_psnr /= max(1, len(test_loader))
        test_ssim /= max(1, len(test_loader))
        test_sam /= max(1, len(test_loader))

        results = {
            "test_l1": round(test_l1, 6),
            "test_psnr": round(test_psnr, 6),
            "test_ssim": round(test_ssim, 6)
        }
        results["test_sam_deg"] = round(test_sam, 6)
        if lpips_model is not None:
            results["test_lpips"] = round(test_lpips / max(1, len(test_loader)), 6)
        if fid_metric is not None:
            results["test_fid"] = round(float(fid_metric.compute()), 6)
        save_json(os.path.join(run_dir, "test_metrics.json"), results)
        logger.info("Test L1: %.6f | PSNR: %.4f | SSIM: %.4f", test_l1, test_psnr, test_ssim)
        logger.info("Test SAM(deg): %.4f", test_sam)
        if lpips_model is not None:
            logger.info("Test LPIPS: %.4f", results["test_lpips"])
        if fid_metric is not None:
            logger.info("Test FID: %.4f", results["test_fid"])


if __name__ == "__main__":
    main()
