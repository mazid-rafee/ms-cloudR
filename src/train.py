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
        "--sar_align_enabled",
        action="store_true",
        default=False,
        help="Enable training-only SAR semantic alignment (DBCR_MR_r3_SARAlign).",
    )
    parser.add_argument(
        "--sar_align_weight",
        type=float,
        default=0.0,
        help="Lambda for L_align. Choose via loss-scale pilot before full runs.",
    )
    parser.add_argument(
        "--sar_align_stage",
        type=int,
        default=3,
        help="Encoder stage for SAR/optical alignment (Exp3 uses 3 only).",
    )
    parser.add_argument(
        "--sar_align_anchor_t",
        type=float,
        default=0.0,
        help="Fixed teacher timestep for clean optical anchor extraction.",
    )
    parser.add_argument(
        "--sar_align_teacher_checkpoint",
        type=str,
        default=(
            "outputs/DBCR_MR_r3_seed42_epochs50_20260812/checkpoints/best.pt"
        ),
        help="Frozen DBCR_MR_r3 teacher checkpoint used only during training.",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        default=False,
        help="Skip end-of-training test evaluation (use for short pilots).",
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
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable_parameters=%d", n_params)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sar_align_enabled = bool(args.sar_align_enabled)
    sar_align_weight = float(args.sar_align_weight)
    sar_align_stage = int(args.sar_align_stage)
    sar_align_anchor_t = float(args.sar_align_anchor_t)
    teacher = None
    param_groups = None
    if sar_align_enabled:
        if sar_align_stage != 3:
            raise ValueError(
                f"Exp3 SARAlign supports stage 3 only, got sar_align_stage={sar_align_stage}"
            )
        from src.utils.sar_align import (
            assert_teacher_frozen,
            extract_detached_optical_anchor,
            feature_representation_stats,
            load_frozen_teacher,
            named_param_checksum,
            optimizer_contains_params,
            parameter_group_tensors,
            spatial_cosine_align_loss,
            state_dict_checksum,
            grad_norm,
        )

        teacher = load_frozen_teacher(
            args.sar_align_teacher_checkpoint,
            device=device,
        )
        assert_teacher_frozen(teacher)
        if optimizer_contains_params(opt, teacher.parameters()):
            raise RuntimeError("Frozen teacher parameters must not enter the optimizer")
        param_groups = parameter_group_tensors(model)
        n_teacher = sum(p.numel() for p in teacher.parameters())
        teacher_checksum_before = state_dict_checksum(teacher)
        student_sar_checksum_before = named_param_checksum(
            model, ("sar_stem", "sar_enc")
        )
        student_downs_checksum_before = named_param_checksum(model, ("downs",))
        logger.info(
            "SARAlign enabled: weight=%s stage=%s anchor_t=%s teacher=%s "
            "teacher_params(train_only_memory)=%d",
            sar_align_weight,
            sar_align_stage,
            sar_align_anchor_t,
            args.sar_align_teacher_checkpoint,
            n_teacher,
        )
        logger.info(
            "SARAlign gradient note: L_align updates student SAR-only params and "
            "shared downs; frozen teacher receives no gradients; optical-only "
            "student params are updated by L_recon only."
        )
        logger.info("teacher_checksum_before=%s", teacher_checksum_before)

    start_epoch = 1
    best_val = None
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer=opt, map_location=device)
        start_epoch = payload.get("epoch", 0) + 1
        best_val = payload.get("best_val")
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    config_payload = {
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
        "sar_align_enabled": sar_align_enabled,
        "sar_align_weight": sar_align_weight,
        "sar_align_stage": sar_align_stage,
        "sar_align_anchor_t": sar_align_anchor_t,
        "sar_align_teacher_checkpoint": (
            args.sar_align_teacher_checkpoint if sar_align_enabled else ""
        ),
        "split_sizes": {
            "train": train_size,
            "val": val_size,
            "test": test_size
        }
    }
    if teacher is not None:
        config_payload["teacher_parameters_train_only_memory"] = sum(
            p.numel() for p in teacher.parameters()
        )
    save_json(os.path.join(run_dir, "config.json"), config_payload)

    metrics_header = ["epoch", "train_loss", "val_loss"]
    if sar_align_enabled:
        metrics_header = [
            "epoch",
            "train_total_loss",
            "train_recon_loss",
            "train_sar_align_loss",
            "train_weighted_sar_align_loss",
            "weighted_align_to_recon_ratio",
            "mean_cosine_similarity",
            "mean_fs_l2",
            "mean_fc_l2",
            "fs_spatial_std",
            "features_finite",
            "grad_norm_sar_only",
            "grad_norm_shared_downs",
            "grad_norm_optical_student",
            "grad_norm_teacher",
            "val_random_t_l1",
            "val_endpoint_l1",
        ]

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if teacher is not None:
            teacher.eval()
        logger.info("Epoch %d/%d", epoch, args.epochs)
        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_align_sum = 0.0
        train_walign_sum = 0.0
        ratio_sum = 0.0
        ratio_count = 0
        cos_sum = 0.0
        fs_l2_sum = 0.0
        fc_l2_sum = 0.0
        fs_std_sum = 0.0
        features_finite_all = True
        last_grad_sar = ""
        last_grad_downs = ""
        last_grad_opt = ""
        last_grad_teacher = ""
        nan_inf_seen = False
        for step, (y, z, x0) in enumerate(train_loader, start=1):
            y = y.to(device)
            z = z.to(device)
            x0 = x0.to(device)

            t = torch.randint(0, args.diffusion_steps + 1, (x0.size(0),), device=device)
            alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
            # Bridge mix uses x0 as endpoint only; student optical path never sees raw x0.
            x_t = (1 - alpha_t) * x0 + alpha_t * y

            if sar_align_enabled:
                x0_hat, aux = model(x_t, t, z, return_aux=True)
                fs = aux["sar_stage3"]
                fc = extract_detached_optical_anchor(
                    teacher, x0, anchor_t=sar_align_anchor_t
                )
                l_recon = torch.mean(torch.abs(x0_hat - x0))
                l_align = spatial_cosine_align_loss(fs, fc)
                l_walign = sar_align_weight * l_align
                loss = l_recon + l_walign
                rep = feature_representation_stats(fs, fc)
                cos_sum += rep["mean_cosine_similarity"]
                fs_l2_sum += rep["mean_fs_l2"]
                fc_l2_sum += rep["mean_fc_l2"]
                fs_std_sum += rep["fs_spatial_std"]
                features_finite_all = features_finite_all and rep["features_finite"]
            else:
                x0_hat = model(x_t, t, z)
                l_recon = torch.mean(torch.abs(x0_hat - x0))
                l_align = None
                l_walign = None
                loss = l_recon

            if not torch.isfinite(loss):
                nan_inf_seen = True

            opt.zero_grad()
            if teacher is not None:
                for p in teacher.parameters():
                    if p.grad is not None:
                        p.grad = None
            loss.backward()
            if sar_align_enabled and param_groups is not None:
                last_grad_sar = round(grad_norm(param_groups["sar_only"]), 6)
                last_grad_downs = round(grad_norm(param_groups["shared_downs"]), 6)
                last_grad_opt = round(grad_norm(param_groups["optical_student"]), 6)
                last_grad_teacher = round(grad_norm(list(teacher.parameters())), 6)
            opt.step()

            train_total_sum += loss.item()
            train_recon_sum += l_recon.item()
            if l_align is not None:
                train_align_sum += l_align.item()
                train_walign_sum += l_walign.item()
                recon_val = max(l_recon.item(), 1e-12)
                ratio_sum += (l_walign.item() / recon_val)
                ratio_count += 1
            if step % args.log_every == 0 or step == len(train_loader):
                progress_bar(f"Train {epoch}", step, len(train_loader))

        sys.stdout.write("\n")
        n_train = max(1, len(train_loader))
        train_loss = train_total_sum / n_train
        train_recon = train_recon_sum / n_train
        train_align = train_align_sum / n_train if sar_align_enabled else None
        train_walign = train_walign_sum / n_train if sar_align_enabled else None
        align_ratio = (ratio_sum / max(1, ratio_count)) if sar_align_enabled else None
        mean_cos = cos_sum / n_train if sar_align_enabled else None
        mean_fs_l2 = fs_l2_sum / n_train if sar_align_enabled else None
        mean_fc_l2 = fc_l2_sum / n_train if sar_align_enabled else None
        mean_fs_std = fs_std_sum / n_train if sar_align_enabled else None

        val_loss = None
        val_endpoint = None
        if len(val_loader) > 0:
            model.eval()
            val_total = 0.0
            val_endpoint_total = 0.0
            with torch.no_grad():
                for step, (y, z, x0) in enumerate(val_loader, start=1):
                    y = y.to(device)
                    z = z.to(device)
                    x0 = x0.to(device)

                    t = torch.randint(0, args.diffusion_steps + 1, (x0.size(0),), device=device)
                    alpha_t = alpha_fn(t.float(), args.diffusion_steps).view(-1, 1, 1, 1)
                    x_t = (1 - alpha_t) * x0 + alpha_t * y
                    # Validation / checkpoint criterion remains reconstruction L1 only.
                    x0_hat = model(x_t, t, z)
                    loss = torch.mean(torch.abs(x0_hat - x0))
                    val_total += loss.item()

                    if sar_align_enabled:
                        t_end = torch.full(
                            (x0.size(0),),
                            float(args.diffusion_steps),
                            device=device,
                        )
                        x0_hat_end = model(y, t_end, z)
                        val_endpoint_total += torch.mean(
                            torch.abs(x0_hat_end - x0)
                        ).item()

                    if step % args.log_every == 0 or step == len(val_loader):
                        progress_bar(f"Val {epoch}", step, len(val_loader))
            sys.stdout.write("\n")
            val_loss = val_total / max(1, len(val_loader))
            if sar_align_enabled:
                val_endpoint = val_endpoint_total / max(1, len(val_loader))

        if sar_align_enabled:
            metrics_row = {
                "epoch": epoch,
                "train_total_loss": round(train_loss, 6),
                "train_recon_loss": round(train_recon, 6),
                "train_sar_align_loss": round(train_align, 6),
                "train_weighted_sar_align_loss": round(train_walign, 6),
                "weighted_align_to_recon_ratio": round(align_ratio, 6),
                "mean_cosine_similarity": round(mean_cos, 6),
                "mean_fs_l2": round(mean_fs_l2, 6),
                "mean_fc_l2": round(mean_fc_l2, 6),
                "fs_spatial_std": round(mean_fs_std, 6),
                "features_finite": bool(features_finite_all and not nan_inf_seen),
                "grad_norm_sar_only": last_grad_sar,
                "grad_norm_shared_downs": last_grad_downs,
                "grad_norm_optical_student": last_grad_opt,
                "grad_norm_teacher": last_grad_teacher,
                "val_random_t_l1": round(val_loss, 6) if val_loss is not None else "",
                "val_endpoint_l1": (
                    round(val_endpoint, 6) if val_endpoint is not None else ""
                ),
            }
            append_metrics_csv(metrics_csv, metrics_row, header=metrics_header)
            logger.info(
                "Train total=%.6f recon=%.6f align=%.6f w_align=%.6f ratio=%.6f "
                "cos=%.6f fs_l2=%.6f fc_l2=%.6f fs_std=%.6f finite=%s "
                "grad_sar=%.6f grad_downs=%.6f grad_opt=%.6f grad_teacher=%.6f",
                train_loss,
                train_recon,
                train_align,
                train_walign,
                align_ratio,
                mean_cos,
                mean_fs_l2,
                mean_fc_l2,
                mean_fs_std,
                features_finite_all and not nan_inf_seen,
                float(last_grad_sar) if last_grad_sar != "" else 0.0,
                float(last_grad_downs) if last_grad_downs != "" else 0.0,
                float(last_grad_opt) if last_grad_opt != "" else 0.0,
                float(last_grad_teacher) if last_grad_teacher != "" else 0.0,
            )
            if val_loss is not None:
                logger.info(
                    "Val random_t_l1=%.6f endpoint_l1=%.6f",
                    val_loss,
                    val_endpoint if val_endpoint is not None else float("nan"),
                )
        else:
            metrics_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else ""
            }
            append_metrics_csv(metrics_csv, metrics_row, header=metrics_header)
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
                    "sar_align_enabled": sar_align_enabled,
                    "sar_align_weight": sar_align_weight,
                }
            )

    logger.info("Training complete.")

    if sar_align_enabled and teacher is not None:
        assert_teacher_frozen(teacher)
        teacher_checksum_after = state_dict_checksum(teacher)
        student_sar_checksum_after = named_param_checksum(
            model, ("sar_stem", "sar_enc")
        )
        student_downs_checksum_after = named_param_checksum(model, ("downs",))
        audit = {
            "teacher_checksum_before": teacher_checksum_before,
            "teacher_checksum_after": teacher_checksum_after,
            "teacher_weights_unchanged": teacher_checksum_before == teacher_checksum_after,
            "teacher_eval_mode": (not teacher.training),
            "teacher_requires_grad_any": any(
                p.requires_grad for p in teacher.parameters()
            ),
            "student_sar_checksum_before": student_sar_checksum_before,
            "student_sar_checksum_after": student_sar_checksum_after,
            "student_sar_changed": student_sar_checksum_before != student_sar_checksum_after,
            "student_downs_checksum_before": student_downs_checksum_before,
            "student_downs_checksum_after": student_downs_checksum_after,
            "student_downs_changed": (
                student_downs_checksum_before != student_downs_checksum_after
            ),
            "x0_never_enters_student_prediction_path": True,
            "note_x0_usage": (
                "x0 used only for bridge mix endpoint, L1 target, and frozen "
                "teacher optical-anchor extraction; student forward always "
                "receives x_t (or y at endpoint val), t, z."
            ),
        }
        save_json(os.path.join(run_dir, "sar_align_pilot_audit.json"), audit)
        logger.info("SARAlign pilot audit: %s", audit)

    if args.skip_test:
        logger.info("Skipping test evaluation (--skip_test).")
        return

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
