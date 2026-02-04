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

        y = self._read(s2c_path)   # (13, H, W) cloudy
        z = self._read(s1_path)    # (2, H, W) SAR
        x0 = self._read(s2_path)   # (13, H, W) clean

        # Optical preprocessing: clip to [0, 10000] and scale to [0, 1].
        y = torch.clamp(y, 0.0, 10000.0) / 10000.0
        x0 = torch.clamp(x0, 0.0, 10000.0) / 10000.0

        # SAR preprocessing: VV in [-25, 0], VH in [-32.5, 0], scale to [0, 1].
        if z.shape[0] >= 2:
            vv = torch.clamp(z[0], -25.0, 0.0)
            vh = torch.clamp(z[1], -32.5, 0.0)
            vv = (vv + 25.0) / 25.0
            vh = (vh + 32.5) / 32.5
            z = torch.stack([vv, vh], dim=0)

        return y, z, x0
