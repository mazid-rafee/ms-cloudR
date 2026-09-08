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


def save_gray_png(path, array, *, vmin=0.0, vmax=1.0):
    try:
        from PIL import Image
    except ImportError:
        return False
    ensure_dir(os.path.dirname(path))
    arr = _to_numpy(array).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    scale = max(float(vmax - vmin), 1e-12)
    arr = np.clip((arr - vmin) / scale, 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(path)
    return True


def save_diverging_png(path, array, *, abs_max=None):
    try:
        from PIL import Image
    except ImportError:
        return False
    ensure_dir(os.path.dirname(path))
    arr = _to_numpy(array).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if abs_max is None:
        abs_max = float(np.max(np.abs(arr))) if arr.size else 1.0
    abs_max = max(abs_max, 1e-12)
    norm = np.clip(arr / abs_max, -1.0, 1.0)
    rgb = np.ones(arr.shape + (3,), dtype=np.float64)
    pos = norm > 0
    neg = norm < 0
    rgb[pos, 1] = 1.0 - norm[pos]
    rgb[pos, 2] = 1.0 - norm[pos]
    rgb[neg, 0] = 1.0 + norm[neg]
    rgb[neg, 1] = 1.0 + norm[neg]
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)
    return True
