import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument(
        "--seasons",
        type=str,
        default="ROIs2017_winter,ROIs1868_summer,ROIs1970_fall,ROIs1158_spring"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def parse_seasons(value):
    return [s.strip() for s in value.split(",") if s.strip()]


def progress_bar(prefix, step, total, bar_width=30):
    filled = int(bar_width * step / total)
    bar = "=" * filled + "." * (bar_width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {step}/{total}")
    sys.stdout.flush()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.data import DataLoader, random_split
    from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
    from src.models.unet_diffusion import UNet
    from src.utils.diffusion import Diffusion
    from src.utils.checkpoint import load_checkpoint
    from src.utils.io_utils import save_json
    from src.utils.logger import setup_logger
    from src.utils.metrics import mse, mae

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
    val_size = int(args.val_split * total)
    test_size = int(args.test_split * total)
    train_size = total - val_size - test_size
    if train_size <= 0:
        logger.error("Invalid split sizes. Adjust val/test splits.")
        return

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

    model = UNet(in_channels=28).to(device)
    diffusion = Diffusion(T=args.diffusion_steps, device=device)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)

    model.eval()
    test_mse = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for step, (x, s1, y) in enumerate(test_loader, start=1):
            x = x.to(device)
            s1 = s1.to(device)
            y = y.to(device)

            r = y - x
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            noise = torch.randn_like(r)
            r_t = diffusion.q_sample(r, t, noise)
            cond = torch.cat([r_t, x, s1], dim=1)
            noise_hat = model(cond)

            test_mse += mse(noise_hat, noise).item()
            test_mae += mae(noise_hat, noise).item()

            if step % args.log_every == 0 or step == len(test_loader):
                progress_bar("Eval", step, len(test_loader))

    sys.stdout.write("\n")
    test_mse /= max(1, len(test_loader))
    test_mae /= max(1, len(test_loader))
    results = {"eval_mse": round(test_mse, 6), "eval_mae": round(test_mae, 6)}
    save_json(os.path.join(run_dir, "eval_metrics.json"), results)
    logger.info("Eval MSE: %.6f | Eval MAE: %.6f", test_mse, test_mae)


if __name__ == "__main__":
    main()
