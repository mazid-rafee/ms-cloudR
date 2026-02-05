import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dbcr import DBCRNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = DBCRNet().to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {params:,}")
    print(f"Trainable: {trainable:,}")

    try:
        from thop import profile
        x_t = torch.zeros(args.batch_size, 13, args.height, args.width, device=device)
        z = torch.zeros(args.batch_size, 2, args.height, args.width, device=device)
        t = torch.zeros(args.batch_size, device=device)
        macs, _ = profile(model, inputs=(x_t, t, z), verbose=False)
        print(f"MACs: {macs:,}")
    except Exception as exc:
        print(f"MACs: unavailable ({exc})")
        print("Install thop: pip install thop")


if __name__ == "__main__":
    main()
