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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
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
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    from src.utils.io_utils import save_json, utc_timestamp
    from src.utils.logger import setup_logger, append_metrics_csv
    from src.utils.metrics import mse, mae

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

    seasons = parse_seasons(args.seasons)
    logger.info("Building dataset...")
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    logger.info("Dataset size: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("No samples found. Check dataset paths and file naming.")
        return

    total = len(dataset)
    val_size = int(args.val_split * total)
    test_size = int(args.test_split * total)
    train_size = total - val_size - test_size
    if train_size <= 0:
        logger.error("Invalid split sizes. Adjust train/val/test splits.")
        return

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

    model = UNet(in_channels=28).to(device)
    diffusion = Diffusion(T=args.diffusion_steps, device=device)
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
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "seed": args.seed,
            "run_name": run_name
        }
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        logger.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = 0.0
        for step, (x, s1, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            s1 = s1.to(device)
            y = y.to(device)

            r = y - x
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            noise = torch.randn_like(r)

            r_t = diffusion.q_sample(r, t, noise)
            cond = torch.cat([r_t, x, s1], dim=1)
            noise_hat = model(cond)
            loss = mse(noise_hat, noise)

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
                for step, (x, s1, y) in enumerate(val_loader, start=1):
                    x = x.to(device)
                    s1 = s1.to(device)
                    y = y.to(device)

                    r = y - x
                    t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
                    noise = torch.randn_like(r)
                    r_t = diffusion.q_sample(r, t, noise)
                    cond = torch.cat([r_t, x, s1], dim=1)
                    noise_hat = model(cond)
                    loss = mse(noise_hat, noise)
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

        if epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
                model,
                opt,
                epoch,
                extra={"best_val": best_val}
            )

        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                model,
                opt,
                epoch,
                extra={"best_val": best_val}
            )

    logger.info("Training complete.")

    if len(test_loader) > 0:
        logger.info("Starting test...")
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
                    progress_bar("Test", step, len(test_loader))
        sys.stdout.write("\n")
        test_mse /= max(1, len(test_loader))
        test_mae /= max(1, len(test_loader))

        results = {"test_mse": round(test_mse, 6), "test_mae": round(test_mae, 6)}
        save_json(os.path.join(run_dir, "test_metrics.json"), results)
        logger.info("Test MSE: %.6f | Test MAE: %.6f", test_mse, test_mae)


if __name__ == "__main__":
    main()
