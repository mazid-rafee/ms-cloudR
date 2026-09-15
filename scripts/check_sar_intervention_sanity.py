#!/usr/bin/env python3
"""Standalone sanity checks for the SAR-intervention diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.sar_intervention_checks import run_sanity_checks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/DBCR_MR_r3_seed42_epochs50_20260812/checkpoints/best.pt",
    )
    parser.add_argument("--seasons", type=str, default="winter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sar_intervention_seed", type=int, default=123)
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--subset_max", type=int, default=16)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--with_model", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/DBCR_MR_r3_SAR_intervention/sanity_checks.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    report = run_sanity_checks(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    checks = report["checks"]
    print(f"Wrote {args.output}")
    for key, value in checks.items():
        mark = "PASS" if value else ("SKIP" if value is None else "FAIL")
        print(f"  [{mark}] {key}: {value}")
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()
