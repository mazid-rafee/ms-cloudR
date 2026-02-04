import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.utils.io_utils import map_seasons


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/SEN12MS-CR")
    parser.add_argument("--seasons", type=str, default="summer")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/invalid_files.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    seasons = map_seasons(args.seasons)
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    if len(dataset) == 0:
        print("No samples found.")
        return

    invalid = []
    count = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
    for idx in range(count):
        y, z, x0 = dataset[idx]
        if not (torch.isfinite(y).all() and torch.isfinite(z).all() and torch.isfinite(x0).all()):
            s2c, s1, s2 = dataset.samples[idx]
            invalid.append((idx, s2c, s1, s2))
        if (idx + 1) % 100 == 0 or (idx + 1) == count:
            print(f"\rScanning {idx + 1}/{count}", end="")
    print()

    print(f"Scanned {count} samples, invalid: {len(invalid)}")
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for idx, s2c, s1, s2 in invalid:
                f.write(f"{idx}\t{s2c}\t{s1}\t{s2}\n")
        print(f"Wrote invalid list to {args.output}")
    for idx, s2c, s1, s2 in invalid[:50]:
        print(f"{idx}\t{s2c}\t{s1}\t{s2}")
    if len(invalid) > 50:
        print(f"... {len(invalid) - 50} more")


if __name__ == "__main__":
    main()
