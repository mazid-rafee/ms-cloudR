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
        choices=[
            "original",
            "mean_reverting",
            "mr_r3",
            "spatial_mr_r3",
            "spatial_mean_reverting",
        ],
        help=(
            "DB-CR bridge: original sinusoidal, scalar MR_r3 (mean_reverting/mr_r3), "
            "or spatially adaptive MR (spatial_mr_r3)."
        ),
    )
    parser.add_argument(
        "--mean_reversion_rate",
        type=float,
        default=3.0,
        help="r / r_max for MR schedules (scalar or SpatialMR).",
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
    from src.utils.spatial_bridge import (
        construct_spatial_mr_bridge,
        is_spatial_mr_schedule,
        normalize_bridge_schedule,
    )

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
    bridge_key = normalize_bridge_schedule(args.bridge_schedule)
    use_spatial_mr = is_spatial_mr_schedule(args.bridge_schedule)
    logger.info(
        "bridge_schedule=%s (normalized=%s) mean_reversion_rate=%s diffusion_steps=%s nfe=%s",
        args.bridge_schedule,
        bridge_key,
        args.mean_reversion_rate,
        args.diffusion_steps,
        args.nfe,
    )
    if use_spatial_mr and int(args.nfe) != 1:
        logger.warning(
            "spatial_mr_r3 is validated for NFE=1 only; got nfe=%s. "
            "Proceeding, but multi-step spatial reverse is NOT implemented.",
            args.nfe,
        )
    # Inference ODE alphas: original / scalar MR_r3 (spatial training still uses spatial A).
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
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable_parameters=%d", n_params)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_val = None
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
            if use_spatial_mr:
                x_t, _, _, _ = construct_spatial_mr_bridge(
                    x0,
                    y,
                    t,
                    args.diffusion_steps,
                    r_max=args.mean_reversion_rate,
                )
            else:
                alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
                x_t = (1 - alpha_t) * x0 + alpha_t * y
            x0_hat = model(x_t, t, z)
            loss = torch.mean(torch.abs(x0_hat - x0))

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
                    if use_spatial_mr:
                        x_t, _, _, _ = construct_spatial_mr_bridge(
                            x0,
                            y,
                            t,
                            args.diffusion_steps,
                            r_max=args.mean_reversion_rate,
                        )
                    else:
                        alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
                        x_t = (1 - alpha_t) * x0 + alpha_t * y
                    x0_hat = model(x_t, t, z)
                    loss = torch.mean(torch.abs(x0_hat - x0))
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
                }
            )

    # Always keep a final-epoch checkpoint (does not replace best.pt).
    save_checkpoint(
        os.path.join(ckpt_dir, "last.pt"),
        model,
        opt,
        args.epochs if start_epoch <= args.epochs else start_epoch - 1,
        extra={
            "best_val": best_val,
            "bridge_schedule": args.bridge_schedule,
            "mean_reversion_rate": args.mean_reversion_rate,
            "checkpoint_type": "last",
        },
    )
    logger.info("Training complete. Saved last.pt and best.pt (if improved).")

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
