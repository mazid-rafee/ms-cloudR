import os
import sys

import numpy as np
import rasterio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PATHS = [
    "data/SEN12MS-CR/ROIs1868_summer_s2_cloudy/s2_cloudy_146/ROIs1868_summer_s2_cloudy_146_p202.tif",
    "data/SEN12MS-CR/ROIs1868_summer_s1/s1_146/ROIs1868_summer_s1_146_p202.tif",
    "data/SEN12MS-CR/ROIs1868_summer_s2/s2_146/ROIs1868_summer_s2_146_p202.tif",
]


def summarize(path):
    with rasterio.open(path) as f:
        data = f.read()
    nan = np.isnan(data).any()
    posinf = np.isposinf(data).any()
    neginf = np.isneginf(data).any()
    summary = {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "nan": bool(nan),
        "posinf": bool(posinf),
        "neginf": bool(neginf),
    }
    if nan:
        idx = np.argwhere(np.isnan(data))
        summary["nan_count"] = int(idx.shape[0])
        summary["nan_first"] = idx[0].tolist()
    if posinf or neginf:
        idx = np.argwhere(~np.isfinite(data))
        summary["inf_count"] = int(idx.shape[0])
        summary["inf_first"] = idx[0].tolist()
    return summary


def main():
    for path in PATHS:
        print(path)
        summary = summarize(path)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
