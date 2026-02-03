import argparse
import sys


def _strip_arg(argv, name):
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == name:
            skip_next = True
            continue
        if arg.startswith(f"{name}="):
            continue
        cleaned.append(arg)
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    args, _ = parser.parse_known_args()

    sys.argv = _strip_arg(sys.argv, "--mode")

    if args.mode == "train":
        from src import train
        train.main()
    else:
        from src import eval as eval_module
        eval_module.main()


if __name__ == "__main__":
    main()
