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
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--spectral_mean_reversion_rates",
        type=float,
        nargs=13,
        default=None,
        help=(
            "Optional 13 band-wise mean-reversion rates for a spectrally "
            "anisotropic mean-reverting bridge. When set, overrides "
            "--mean_reversion_rate. Requires --bridge_schedule mean_reverting."
        ),
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--num_save_images", type=int, default=4)
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--subset_max", type=int, default=0)
    parser.add_argument(
        "--sar_intervention",
        type=str,
        default="normal",
        choices=["normal", "zero", "shuffled", "noise", "all"],
        help=(
            "Inference-only SAR-branch intervention. Default 'normal' is identical "
            "to previous DBCR_MR_r3 eval. 'all' is handled by "
            "scripts/eval_sar_intervention.py."
        ),
    )
    parser.add_argument(
        "--sar_intervention_seed",
        type=int,
        default=123,
        help="Seed for shuffled derangement and noise SAR generation.",
    )
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="Write per-sample metrics.csv for later paired intervention analysis.",
    )
    parser.add_argument(
        "--region_metrics",
        action="store_true",
        help=(
            "Compute eval-only soft cloud/clear L1 and SAM using the SpatialMR "
            "soft cloud score. The mask is never fed to the model."
        ),
    )
    parser.add_argument(
        "--write_intervention_artifacts",
        action="store_true",
        help="Write metrics.json plus shuffle/noise artifacts for the SAR study.",
    )
    parser.add_argument(
        "--intervention_export_dir",
        type=str,
        default="",
        help="Optional parent directory for shuffle mapping / noise statistics JSON.",
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


def _maybe_wrap_sar_intervention(test_ds, args, logger, run_dir):
    """Wrap the test split only when an intervention or extra artifacts are requested.

    Default --sar_intervention normal with no extra flags leaves test_ds unchanged.
    """
    from src.utils.sar_intervention import (
        SARInterventionDataset,
        build_shuffle_mapping,
        estimate_processed_sar_stats,
        normalize_intervention,
        unwrap_subset,
    )

    mode = normalize_intervention(args.sar_intervention)
    if mode == "all":
        raise ValueError(
            "--sar_intervention all is orchestrated by "
            "scripts/eval_sar_intervention.py, not by a single eval loop."
        )
    wrap = (
        mode != "normal"
        or bool(args.save_per_sample)
        or bool(args.write_intervention_artifacts)
    )
    if not wrap:
        return test_ds, None, None

    root, test_indices = unwrap_subset(test_ds)
    perm = None
    mapping = None
    noise_stats = None
    from src.utils.io_utils import save_json
    if mode == "shuffled":
        perm, mapping = build_shuffle_mapping(
            root, test_indices, seed=args.sar_intervention_seed
        )
        mapping_name = f"sar_shuffle_mapping_seed{args.sar_intervention_seed}.json"
        save_json(os.path.join(run_dir, mapping_name), mapping)
        export_dir = getattr(args, "intervention_export_dir", "") or ""
        if export_dir:
            save_json(os.path.join(export_dir, mapping_name), mapping)
        logger.info(
            "SAR shuffle derangement n=%d seed=%s fixed_points=0",
            len(perm),
            args.sar_intervention_seed,
        )
    if mode == "noise":
        noise_stats = estimate_processed_sar_stats(root, test_indices)
        save_json(os.path.join(run_dir, "sar_noise_statistics.json"), noise_stats)
        export_dir = getattr(args, "intervention_export_dir", "") or ""
        if export_dir:
            save_json(os.path.join(export_dir, "sar_noise_statistics.json"), noise_stats)
        logger.info(
            "SAR noise stats VV mean/std=%.6f/%.6f VH mean/std=%.6f/%.6f",
            noise_stats["channels"]["VV"]["mean"],
            noise_stats["channels"]["VV"]["std"],
            noise_stats["channels"]["VH"]["mean"],
            noise_stats["channels"]["VH"]["std"],
        )

    wrapped = SARInterventionDataset(
        root,
        test_indices,
        mode,
        perm=perm if mode == "shuffled" else None,
        noise_stats=noise_stats if mode == "noise" else None,
        noise_seed=args.sar_intervention_seed,
    )
    logger.info("SAR intervention: %s (wrapped test set, n=%d)", mode, len(wrapped))
    return wrapped, mapping, noise_stats


def main():
    args, defaults = parse_args()
    results = run_eval(args, defaults)
    return results


def run_eval(args, defaults=None):
    if args.config:
        from src.utils.config import load_config, apply_config
        args = apply_config(args, load_config(args.config), defaults=defaults)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.data import DataLoader, random_split, Subset
    from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
    from src.models.dbcr import get_alpha_schedule, reshape_alpha_for_broadcast
    from src.models.registry import get_model
    from src.utils.checkpoint import load_checkpoint
    from src.utils.io_utils import save_json
    from src.utils.logger import setup_logger
    from src.utils.metrics import psnr, ssim, sam_deg
    from src.utils.image_utils import save_npy, save_rgb_png

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.output_dir, args.run_name)
    log_dir = os.path.join(run_dir, "logs")
    logger = setup_logger(log_dir, "eval")
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("seed=%s sar_intervention_seed=%s", args.seed, args.sar_intervention_seed)
    spectral_rates = args.spectral_mean_reversion_rates
    if spectral_rates is not None:
        spectral_rates = [float(r) for r in spectral_rates]
    logger.info(
        "Bridge schedule: %s (mean_reversion_rate=%s spectral_mean_reversion_rates=%s) "
        "nfe=%s sar_intervention=%s",
        args.bridge_schedule,
        args.mean_reversion_rate,
        spectral_rates,
        args.nfe,
        args.sar_intervention,
    )
    logger.info("checkpoint=%s", args.checkpoint)
    alpha_fn = get_alpha_schedule(
        bridge_schedule=args.bridge_schedule,
        mean_reversion_rate=args.mean_reversion_rate,
        spectral_mean_reversion_rates=spectral_rates,
    )

    seasons = parse_seasons(args.seasons)
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    if len(dataset) == 0:
        logger.error("No samples found. Check dataset paths and file naming.")
        return None
    if args.subset_frac < 1.0 or args.subset_max > 0:
        max_len = len(dataset)
        frac_len = max(1, int(max_len * args.subset_frac))
        if args.subset_max > 0:
            frac_len = min(frac_len, args.subset_max)
        g_subset = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(max_len, generator=g_subset)[:frac_len]
        dataset = Subset(dataset, indices.tolist())

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    _, _, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    test_ds, shuffle_mapping, noise_stats = _maybe_wrap_sar_intervention(
        test_ds, args, logger, run_dir
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    model_cls = get_model(args.model)
    if args.model != "dbcr":
        raise NotImplementedError("Only DB-CR is wired into the eval loop right now.")
    model = model_cls().to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)

    model.eval()
    test_l1 = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    test_sam = 0.0
    test_lpips = 0.0
    test_cloud_l1 = 0.0
    test_clear_l1 = 0.0
    test_cloud_sam = 0.0
    test_clear_sam = 0.0
    saved = 0
    per_sample_rows = []
    lpips_model = None
    fid_metric = None
    cloud_score_fn = None
    if args.region_metrics:
        from src.utils.cloud_score import compute_soft_cloud_score
        from src.utils.sar_intervention import soft_region_l1, soft_region_sam_deg
        cloud_score_fn = compute_soft_cloud_score
        logger.info(
            "Region metrics ON (eval-only). Soft cloud score is NOT fed to the model."
        )
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
    wrapped_batches = hasattr(test_ds, "mode")
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            if wrapped_batches:
                y, z, x0, meta = batch
            else:
                y, z, x0 = batch
                meta = None
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
                alpha_curr = reshape_alpha_for_broadcast(
                    alpha_fn(t_curr.float(), T), as_channels=True
                )
                alpha_next = reshape_alpha_for_broadcast(
                    alpha_fn(t_next.float(), T), as_channels=True
                )
                x0_hat = model(x_t, t_curr.repeat(x_t.size(0)), z)
                x_t = (1 - alpha_next / alpha_curr) * x0_hat + (alpha_next / alpha_curr) * x_t

            test_l1 += torch.mean(torch.abs(x0_hat - x0)).item()
            test_psnr += psnr(x0_hat, x0).item()
            test_ssim += ssim(x0_hat, x0).item()
            test_sam += sam_deg(x0_hat, x0).item()

            cloud_l1_b = None
            clear_l1_b = None
            cloud_sam_b = None
            clear_sam_b = None
            cloud_score = None
            if cloud_score_fn is not None:
                cloud_score = cloud_score_fn(y)
                cloud_l1_b = soft_region_l1(x0_hat, x0, cloud_score)
                clear_l1_b = soft_region_l1(x0_hat, x0, 1.0 - cloud_score)
                cloud_sam_b = soft_region_sam_deg(x0_hat, x0, cloud_score)
                clear_sam_b = soft_region_sam_deg(x0_hat, x0, 1.0 - cloud_score)
                test_cloud_l1 += float(cloud_l1_b)
                test_clear_l1 += float(clear_l1_b)
                test_cloud_sam += float(cloud_sam_b)
                test_clear_sam += float(clear_sam_b)

            lpips_batch = None
            if lpips_model is not None or fid_metric is not None:
                rgb_pred = x0_hat[:, [3, 2, 1], :, :].clamp(0.0, 1.0)
                rgb_gt = x0[:, [3, 2, 1], :, :].clamp(0.0, 1.0)
                if lpips_model is not None:
                    lp_pred = rgb_pred * 2.0 - 1.0
                    lp_gt = rgb_gt * 2.0 - 1.0
                    lpips_batch = lpips_model(lp_pred, lp_gt)
                    test_lpips += lpips_batch.mean().item()
                if fid_metric is not None:
                    rgb_pred_u8 = (rgb_pred * 255.0).to(torch.uint8)
                    rgb_gt_u8 = (rgb_gt * 255.0).to(torch.uint8)
                    fid_metric.update(rgb_pred_u8, real=False)
                    fid_metric.update(rgb_gt_u8, real=True)

            if args.save_per_sample:
                bsz = x0_hat.size(0)
                for i in range(bsz):
                    row = {
                        "sample_id": "",
                        "test_index": "",
                        "dataset_index": "",
                        "sar_intervention": args.sar_intervention,
                        "sar_source_sample_id": "",
                        "L1": float(torch.mean(torch.abs(x0_hat[i] - x0[i])).item()),
                        "PSNR": float(psnr(x0_hat[i:i + 1], x0[i:i + 1]).item()),
                        "SSIM": float(ssim(x0_hat[i:i + 1], x0[i:i + 1]).item()),
                        "SAM": float(sam_deg(x0_hat[i:i + 1], x0[i:i + 1]).item()),
                    }
                    if meta is not None:
                        row["sample_id"] = meta["sample_id"][i]
                        row["test_index"] = int(meta["test_index"][i])
                        row["dataset_index"] = int(meta["dataset_index"][i])
                        row["sar_source_sample_id"] = meta["sar_source_sample_id"][i]
                    if lpips_batch is not None:
                        row["LPIPS"] = float(lpips_batch[i].mean().item())
                    if cloud_score is not None:
                        row["cloud_L1_soft"] = float(
                            soft_region_l1(
                                x0_hat[i:i + 1], x0[i:i + 1], cloud_score[i:i + 1]
                            ).item()
                        )
                        row["clear_L1_soft"] = float(
                            soft_region_l1(
                                x0_hat[i:i + 1],
                                x0[i:i + 1],
                                1.0 - cloud_score[i:i + 1],
                            ).item()
                        )
                        row["cloud_SAM_soft"] = float(
                            soft_region_sam_deg(
                                x0_hat[i:i + 1], x0[i:i + 1], cloud_score[i:i + 1]
                            ).item()
                        )
                        row["clear_SAM_soft"] = float(
                            soft_region_sam_deg(
                                x0_hat[i:i + 1],
                                x0[i:i + 1],
                                1.0 - cloud_score[i:i + 1],
                            ).item()
                        )
                    per_sample_rows.append(row)

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
                progress_bar("Eval", step, len(test_loader))

    sys.stdout.write("\n")
    test_l1 /= max(1, len(test_loader))
    test_psnr /= max(1, len(test_loader))
    test_ssim /= max(1, len(test_loader))
    test_sam /= max(1, len(test_loader))
    results = {
        "eval_l1": round(test_l1, 6),
        "eval_psnr": round(test_psnr, 6),
        "eval_ssim": round(test_ssim, 6),
        "bridge_schedule": args.bridge_schedule,
        "mean_reversion_rate": args.mean_reversion_rate,
        "spectral_mean_reversion_rates": spectral_rates,
    }
    if args.write_intervention_artifacts or args.sar_intervention != "normal":
        results.update(
            {
                "nfe": args.nfe,
                "seed": args.seed,
                "sar_intervention": args.sar_intervention,
                "sar_intervention_seed": args.sar_intervention_seed,
                "checkpoint": args.checkpoint,
            }
        )
    results["eval_sam_deg"] = round(test_sam, 6)
    if lpips_model is not None:
        results["eval_lpips"] = round(test_lpips / max(1, len(test_loader)), 6)
    if fid_metric is not None:
        results["eval_fid"] = round(float(fid_metric.compute()), 6)
    if cloud_score_fn is not None:
        results["eval_cloud_l1_soft"] = round(test_cloud_l1 / max(1, len(test_loader)), 6)
        results["eval_clear_l1_soft"] = round(test_clear_l1 / max(1, len(test_loader)), 6)
        results["eval_cloud_sam_soft"] = round(test_cloud_sam / max(1, len(test_loader)), 6)
        results["eval_clear_sam_soft"] = round(test_clear_sam / max(1, len(test_loader)), 6)
    save_json(os.path.join(run_dir, "eval_metrics.json"), results)
    if args.write_intervention_artifacts or args.save_per_sample:
        summary_metrics = {
            "L1": results["eval_l1"],
            "PSNR": results["eval_psnr"],
            "SSIM": results["eval_ssim"],
            "SAM": results["eval_sam_deg"],
            "LPIPS": results.get("eval_lpips"),
            "FID": results.get("eval_fid"),
            "cloud_L1_soft": results.get("eval_cloud_l1_soft"),
            "clear_L1_soft": results.get("eval_clear_l1_soft"),
            "cloud_SAM_soft": results.get("eval_cloud_sam_soft"),
            "clear_SAM_soft": results.get("eval_clear_sam_soft"),
            "sar_intervention": args.sar_intervention,
            "bridge_schedule": args.bridge_schedule,
            "mean_reversion_rate": args.mean_reversion_rate,
            "spectral_mean_reversion_rates": spectral_rates,
            "nfe": args.nfe,
            "seed": args.seed,
            "sar_intervention_seed": args.sar_intervention_seed,
            "checkpoint": args.checkpoint,
        }
        save_json(os.path.join(run_dir, "metrics.json"), summary_metrics)
    if per_sample_rows:
        from src.utils.sar_intervention import write_per_sample_csv
        fieldnames = [
            "sample_id",
            "test_index",
            "dataset_index",
            "sar_intervention",
            "sar_source_sample_id",
            "L1",
            "PSNR",
            "SSIM",
            "SAM",
            "LPIPS",
            "cloud_L1_soft",
            "clear_L1_soft",
            "cloud_SAM_soft",
            "clear_SAM_soft",
        ]
        write_per_sample_csv(
            os.path.join(run_dir, "per_sample_metrics.csv"),
            per_sample_rows,
            fieldnames,
        )
    logger.info("Eval L1: %.6f | PSNR: %.4f | SSIM: %.4f", test_l1, test_psnr, test_ssim)
    logger.info("Eval SAM(deg): %.4f", test_sam)
    if lpips_model is not None:
        logger.info("Eval LPIPS: %.4f", results["eval_lpips"])
    if fid_metric is not None:
        logger.info("Eval FID: %.4f", results["eval_fid"])
    if cloud_score_fn is not None:
        logger.info(
            "Eval cloud/clear L1(soft): %.6f / %.6f | SAM(soft): %.4f / %.4f",
            results["eval_cloud_l1_soft"],
            results["eval_clear_l1_soft"],
            results["eval_cloud_sam_soft"],
            results["eval_clear_sam_soft"],
        )
    return results


if __name__ == "__main__":
    main()
