import logging
import os

from .io_utils import ensure_dir


def setup_logger(log_dir, name):
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def append_metrics_csv(path, metrics, header=None):
    ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            keys = header if header is not None else list(metrics.keys())
            f.write(",".join(keys) + "\n")
        values = [str(metrics[k]) for k in (header or metrics.keys())]
        f.write(",".join(values) + "\n")
