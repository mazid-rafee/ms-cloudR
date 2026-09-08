import os
import torch

from .io_utils import ensure_dir


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    ensure_dir(os.path.dirname(path))
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Local experiment checkpoints may contain optimizer/scheduler metadata.
    # Explicitly disable weights_only for trusted project artifacts (PyTorch >=2.6).
    try:
        payload = torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
