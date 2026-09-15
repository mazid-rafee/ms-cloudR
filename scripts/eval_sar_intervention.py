#!/usr/bin/env python3
"""Run DBCR_MR_r3 SAR-intervention diagnostics (inference only).

Default baseline:
  checkpoint = outputs/DBCR_MR_r3_seed42_epochs50_20260812/checkpoints/best.pt
  config     = configs/dbcr_mean_reverting.json
  NFE        = 1
  split seed = 42

Does not train or finetune. Use --sanity_only for the pre-eval checks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_CHECKPOINT = os.path.join(
    "outputs", "DBCR_MR_r3_seed42_epochs50_20260812", "checkpoints", "best.pt"
)
DEFAULT_CONFIG = os.path.join("configs", "dbcr_mean_reverting.json")
DEFAULT_OUTPUT = os.path.join("outputs", "DBCR_MR_r3_SAR_intervention")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seasons", type=str, default="winter,summer,fall,spring")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sar_intervention_seed", type=int, default=123)
    parser.add_argument(
        "--sar_intervention",
        type=str,
        default="all",
        choices=["normal", "zero", "shuffled", "noise", "all"],
    )
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--subset_max", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--num_save_images", type=int, default=4)
    parser.add_argument("--sanity_only", action="store_true")
    parser.add_argument("--with_model", action="store_true")
    return parser.parse_args()


def file_sha256(path: str, nbytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(nbytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_one_mode(args, mode: str):
    from src.eval import run_eval

    eval_args = argparse.Namespace(
        gpu=args.gpu,
        data_dir=args.data_dir,
        model="dbcr",
        config=args.config,
        seasons=args.seasons,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
        nfe=args.nfe,
        bridge_schedule="mean_reverting",
        mean_reversion_rate=3.0,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        run_name=mode,
        log_every=args.log_every,
        save_images=args.save_images,
        num_save_images=args.num_save_images,
        subset_frac=args.subset_frac,
        subset_max=args.subset_max,
        sar_intervention=mode,
        sar_intervention_seed=args.sar_intervention_seed,
        save_per_sample=True,
        region_metrics=True,
        write_intervention_artifacts=True,
        intervention_export_dir=args.output_dir,
    )
    defaults = {
        "gpu": "0",
        "data_dir": "data/SEN12MS-CR",
        "model": "dbcr",
        "config": "",
        "seasons": "winter,summer,fall,spring",
        "batch_size": 4,
        "num_workers": 4,
        "diffusion_steps": 1000,
        "seed": 42,
        "nfe": 1,
        "bridge_schedule": "original",
        "mean_reversion_rate": 3.0,
        "output_dir": "outputs",
        "run_name": "eval",
        "log_every": 10,
        "save_images": False,
        "num_save_images": 4,
        "subset_frac": 1.0,
        "subset_max": 0,
        "sar_intervention": "normal",
        "sar_intervention_seed": 123,
        "save_per_sample": False,
        "region_metrics": False,
        "write_intervention_artifacts": False,
        "intervention_export_dir": "",
    }
    return run_eval(eval_args, defaults=defaults)


def run_sanity(args) -> dict:
    from src.utils.sar_intervention_checks import run_sanity_checks

    sanity_args = argparse.Namespace(**vars(args))
    # Keep pre-eval sanity cheap even when a later full test is requested.
    if int(getattr(args, "subset_max", 0) or 0) <= 0:
        sanity_args.subset_max = 16
        if "," in str(args.seasons):
            sanity_args.seasons = str(args.seasons).split(",")[0]
    return run_sanity_checks(sanity_args)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.nfe != 1:
        print(
            "WARNING: this diagnostic is specified for NFE=1; "
            f"received nfe={args.nfe}",
            file=sys.stderr,
        )
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    sanity = run_sanity(args)
    from src.utils.io_utils import save_json
    save_json(os.path.join(args.output_dir, "sanity_checks.json"), sanity)
    failed = [k for k, v in sanity["checks"].items() if v is False]
    if failed:
        raise SystemExit(f"Sanity checks failed: {failed}")
    print("Sanity checks passed.")
    if args.sanity_only:
        return

    modes = (
        ["normal", "zero", "shuffled", "noise"]
        if args.sar_intervention == "all"
        else [args.sar_intervention]
    )
    per_mode = {}
    for mode in modes:
        print(f"=== SAR intervention: {mode} ===")
        results = run_one_mode(args, mode)
        if results is None:
            raise RuntimeError(f"Eval returned no results for mode={mode}")
        per_mode[mode] = {
            "L1": results.get("eval_l1"),
            "PSNR": results.get("eval_psnr"),
            "SSIM": results.get("eval_ssim"),
            "SAM": results.get("eval_sam_deg"),
            "LPIPS": results.get("eval_lpips"),
            "FID": results.get("eval_fid"),
            "cloud_L1_soft": results.get("eval_cloud_l1_soft"),
            "clear_L1_soft": results.get("eval_clear_l1_soft"),
            "cloud_SAM_soft": results.get("eval_cloud_sam_soft"),
            "clear_SAM_soft": results.get("eval_clear_sam_soft"),
        }

    from src.utils.sar_intervention import write_summary_tables

    extras = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config": args.config,
        "bridge_schedule": "mean_reverting",
        "mean_reversion_rate": 3.0,
        "nfe": args.nfe,
        "seed": args.seed,
        "sar_intervention_seed": args.sar_intervention_seed,
    }
    write_summary_tables(args.output_dir, per_mode, extras=extras)
    print(f"Wrote summary to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
