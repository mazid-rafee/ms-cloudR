import os
import rasterio
import torch
from torch.utils.data import Dataset

class SEN12MSCRDataset(Dataset):
    def __init__(self, base_dir, seasons):
        self.samples = []
        valid_exts = {".tif", ".tiff"}

        for season in seasons:
            s2c_dir = os.path.join(base_dir, f"{season}_s2_cloudy")
            s2_dir  = os.path.join(base_dir, f"{season}_s2")
            s1_dir  = os.path.join(base_dir, f"{season}_s1")

            for root, _, files in os.walk(s2c_dir):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in valid_exts:
                        continue
                    s2c_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(s2c_path, s2c_dir)
                    s1_rel = rel_path.replace("s2_cloudy_", "s1_")
                    s2_rel = rel_path.replace("s2_cloudy_", "s2_")
                    s1_path = os.path.join(s1_dir, s1_rel)
                    s2_path = os.path.join(s2_dir, s2_rel)
                    if os.path.isfile(s1_path) and os.path.isfile(s2_path):
                        self.samples.append((s2c_path, s1_path, s2_path))

    def __len__(self):
        return len(self.samples)

    def _read(self, path):
        with rasterio.open(path) as f:
            return torch.from_numpy(f.read()).float()

    def __getitem__(self, idx):
        s2c_path, s1_path, s2_path = self.samples[idx]

        x = self._read(s2c_path)   # (13, H, W)
        s1 = self._read(s1_path)   # (2, H, W)
        y = self._read(s2_path)    # (13, H, W)

        return x, s1, y
