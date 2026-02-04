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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--num_save_images", type=int, default=4)
    return parser.parse_args()


def parse_seasons(value):
    from src.utils.io_utils import map_seasons
    return map_seasons(value)


def progress_bar(prefix, step, total, bar_width=30):
    filled = int(bar_width * step / total)
    bar = "=" * filled + "." * (bar_width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {step}/{total}")
    sys.stdout.flush()


def main():
    args = parse_args()
    if args.config:
        from src.utils.config import load_config, apply_config
        args = apply_config(args, load_config(args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.data import DataLoader, random_split
    from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
    from src.models.dbcr import alpha_schedule
    from src.models.registry import get_model
    from src.utils.checkpoint import load_checkpoint
    from src.utils.io_utils import save_json
    from src.utils.logger import setup_logger
    from src.utils.metrics import psnr, ssim
    from src.utils.image_utils import save_npy, save_rgb_png

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.output_dir, args.run_name)
    log_dir = os.path.join(run_dir, "logs")
    logger = setup_logger(log_dir, "eval")
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    seasons = parse_seasons(args.seasons)
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    if len(dataset) == 0:
        logger.error("No samples found. Check dataset paths and file naming.")
        return

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    _, _, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
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
        raise NotImplementedError("Only DB-CR is wired into the eval loop right now.")
    model = model_cls().to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)

    model.eval()
    test_l1 = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    saved = 0
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
                alpha_curr = alpha_schedule(t_curr.float(), T).view(1, 1, 1, 1)
                alpha_next = alpha_schedule(t_next.float(), T).view(1, 1, 1, 1)
                x0_hat = model(x_t, t_curr.repeat(x_t.size(0)), z)
                x_t = (1 - alpha_next / alpha_curr) * x0_hat + (alpha_next / alpha_curr) * x_t

            test_l1 += torch.mean(torch.abs(x0_hat - x0)).item()
            test_psnr += psnr(x0_hat, x0).item()
            test_ssim += ssim(x0_hat, x0).item()

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
    results = {
        "eval_l1": round(test_l1, 6),
        "eval_psnr": round(test_psnr, 6),
        "eval_ssim": round(test_ssim, 6)
    }
    save_json(os.path.join(run_dir, "eval_metrics.json"), results)
    logger.info("Eval L1: %.6f | PSNR: %.4f | SSIM: %.4f", test_l1, test_psnr, test_ssim)


if __name__ == "__main__":
    main()
