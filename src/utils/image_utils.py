import os
import numpy as np

from .io_utils import ensure_dir


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return x


def save_npy(path, array):
    ensure_dir(os.path.dirname(path))
    np.save(path, _to_numpy(array))


def save_rgb_png(path, array, rgb_indices=(3, 2, 1)):
    try:
        from PIL import Image
    except ImportError:
        return False
    ensure_dir(os.path.dirname(path))
    arr = _to_numpy(array)
    if arr.ndim == 3:
        arr = arr[rgb_indices, :, :]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr).save(path)
    return True
