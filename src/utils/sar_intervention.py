"""SAR-branch intervention helpers for DBCR_MR_r3 inference diagnostics.

Interventions are applied AFTER SEN12MSCRDataset preprocessing/normalization
and BEFORE the SAR tensor enters DBCRNet. Optical tensors and the MR_r3
bridge are never modified here.
"""

from __future__ import annotations

import csv
import hashlib
import os
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import Dataset, Subset

from src.utils.io_utils import ensure_dir, save_json


INTERVENTION_MODES = ("normal", "zero", "shuffled", "noise")
LOWER_IS_BETTER = (
    "L1",
    "SAM",
    "LPIPS",
    "FID",
    "cloud_L1_soft",
    "clear_L1_soft",
    "cloud_SAM_soft",
    "clear_SAM_soft",
)
HIGHER_IS_BETTER = ("PSNR", "SSIM")


def normalize_intervention(name: str) -> str:
    mode = str(name).lower().strip()
    if mode not in INTERVENTION_MODES and mode != "all":
        raise ValueError(
            f"Unknown sar_intervention '{name}'. "
            f"Expected one of {INTERVENTION_MODES + ('all',)}."
        )
    return mode


def unwrap_subset(dataset) -> tuple[Any, list[int]]:
    """Unwrap nested Subset/random_split wrappers to (root, global_indices)."""
    chain = []
    ds = dataset
    while isinstance(ds, Subset):
        chain.append(list(ds.indices))
        ds = ds.dataset
    if not chain:
        return ds, list(range(len(ds)))
    composed = list(chain[-1])
    for outer in reversed(chain[:-1]):
        composed = [composed[i] for i in outer]
    return ds, composed


def sattolo_derangement(n: int, seed: int) -> list[int]:
    """Deterministic cyclic derangement (Sattolo). Requires n >= 2.

    Guarantees perm[i] != i for every i. VV/VH stay paired because the
    permutation is over whole samples, not channels.
    """
    if n < 2:
        raise ValueError(f"Derangement requires n >= 2, got n={n}")
    rng = torch.Generator()
    rng.manual_seed(int(seed))
    perm = list(range(n))
    i = n
    while i > 1:
        i -= 1
        j = int(torch.randint(0, i, (1,), generator=rng).item())
        perm[j], perm[i] = perm[i], perm[j]
    if any(perm[k] == k for k in range(n)):
        raise RuntimeError("Sattolo produced a fixed point; this is a bug.")
    return perm


def sample_id_from_root(root_dataset, dataset_index: int) -> str:
    meta = getattr(root_dataset, "sample_meta", None)
    if meta is not None and 0 <= dataset_index < len(meta):
        item = meta[dataset_index]
        return (
            item.get("relative_path")
            or item.get("cloudy_path")
            or str(dataset_index)
        )
    return str(dataset_index)


def sample_meta_from_root(root_dataset, dataset_index: int) -> dict:
    meta = getattr(root_dataset, "sample_meta", None)
    if meta is not None and 0 <= dataset_index < len(meta):
        item = dict(meta[dataset_index])
        item["dataset_index"] = dataset_index
        item["sample_id"] = sample_id_from_root(root_dataset, dataset_index)
        return item
    return {
        "dataset_index": dataset_index,
        "sample_id": str(dataset_index),
    }


def build_shuffle_mapping(
    root_dataset,
    test_indices: list[int],
    seed: int,
) -> tuple[list[int], dict]:
    perm = sattolo_derangement(len(test_indices), seed=seed)
    records = []
    for i, src_ds_idx in enumerate(test_indices):
        j = perm[i]
        dst_ds_idx = test_indices[j]
        records.append(
            {
                "test_index": i,
                "original_dataset_index": int(src_ds_idx),
                "original_sample_id": sample_id_from_root(root_dataset, src_ds_idx),
                "replacement_test_index": int(j),
                "replacement_dataset_index": int(dst_ds_idx),
                "replacement_sar_sample_id": sample_id_from_root(
                    root_dataset, dst_ds_idx
                ),
            }
        )
    payload = {
        "algorithm": "sattolo_cycle",
        "seed": int(seed),
        "n_test": len(test_indices),
        "derangement_verified": all(perm[i] != i for i in range(len(perm))),
        "note": (
            "Dataset-level permutation of processed Sentinel-1 tensors. "
            "VV and VH stay together because each replacement is a full SAR sample."
        ),
        "mapping": records,
    }
    return perm, payload


def estimate_processed_sar_stats(root_dataset, test_indices: Iterable[int]) -> dict:
    """Per-channel mean/std of already-normalized test-set SAR (VV, VH)."""
    sum_c = torch.zeros(2, dtype=torch.float64)
    sumsq_c = torch.zeros(2, dtype=torch.float64)
    count = 0
    n_samples = 0
    for ds_idx in test_indices:
        _, z, _ = root_dataset[int(ds_idx)]
        if z.ndim != 3 or z.shape[0] < 2:
            raise ValueError(f"Expected SAR [2,H,W], got {tuple(z.shape)}")
        z2 = z[:2].reshape(2, -1).to(torch.float64)
        sum_c += z2.sum(dim=1)
        sumsq_c += (z2 * z2).sum(dim=1)
        count += int(z2.shape[1])
        n_samples += 1
    if count <= 0:
        raise ValueError("No SAR pixels found while estimating noise statistics.")
    mean = sum_c / count
    var = torch.clamp(sumsq_c / count - mean * mean, min=0.0)
    std = torch.sqrt(var)
    return {
        "source": "processed_test_set_sar",
        "preprocessing": (
            "VV: clamp(z, -25, 0); (VV + 25) / 25. "
            "VH: clamp(z, -32.5, 0); (VH + 32.5) / 32.5. "
            "Theoretical normalized range is [0, 1] for both channels."
        ),
        "normalized_range": [0.0, 1.0],
        "n_samples": n_samples,
        "n_pixels_per_channel": count,
        "channels": {
            "VV": {
                "index": 0,
                "mean": float(mean[0]),
                "std": float(std[0]),
            },
            "VH": {
                "index": 1,
                "mean": float(mean[1]),
                "std": float(std[1]),
            },
        },
    }


def _stable_int_seed(*parts: int) -> int:
    raw = "|".join(str(int(p)) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def noise_sar_like(z: torch.Tensor, stats: dict, seed: int, sample_key: int) -> torch.Tensor:
    """Replace processed SAR with N(mu_c, sigma_c), then clamp to [0, 1].

    Mean/std and the Generator seed are unchanged. Only the final tensor is
    clipped onto the valid processed-SAR range.
    """
    mu = torch.tensor(
        [stats["channels"]["VV"]["mean"], stats["channels"]["VH"]["mean"]],
        dtype=z.dtype,
    )
    sigma = torch.tensor(
        [stats["channels"]["VV"]["std"], stats["channels"]["VH"]["std"]],
        dtype=z.dtype,
    )
    g = torch.Generator()
    g.manual_seed(_stable_int_seed(seed, sample_key))
    noise = torch.randn(z.shape, generator=g, dtype=z.dtype)
    mu = mu.view(-1, *([1] * (z.ndim - 1)))
    sigma = sigma.view(-1, *([1] * (z.ndim - 1)))
    return (mu + sigma * noise).clamp(0.0, 1.0)


class SARInterventionDataset(Dataset):
    """Wrap a test split so only the SAR tensor is remapped/replaced.

    Optical cloudy S2 (y) and clean S2 (x0) always come from sample i.
    """

    def __init__(
        self,
        root_dataset,
        test_indices: list[int],
        mode: str,
        *,
        perm: Optional[list[int]] = None,
        noise_stats: Optional[dict] = None,
        noise_seed: int = 123,
    ):
        self.root_dataset = root_dataset
        self.test_indices = list(test_indices)
        self.mode = normalize_intervention(mode)
        if self.mode == "all":
            raise ValueError("SARInterventionDataset expects a single mode, not 'all'.")
        self.perm = list(perm) if perm is not None else None
        self.noise_stats = noise_stats
        self.noise_seed = int(noise_seed)
        if self.mode == "shuffled":
            if self.perm is None:
                raise ValueError("shuffled mode requires a dataset-level permutation.")
            if len(self.perm) != len(self.test_indices):
                raise ValueError("Permutation length must match the test set.")
            if any(self.perm[i] == i for i in range(len(self.perm))):
                raise ValueError("Permutation is not a derangement.")
        if self.mode == "noise" and self.noise_stats is None:
            raise ValueError("noise mode requires processed SAR statistics.")

    def __len__(self) -> int:
        return len(self.test_indices)

    def __getitem__(self, i: int):
        src_ds_idx = self.test_indices[i]
        y, z, x0 = self.root_dataset[src_ds_idx]
        sar_source_test_index = i
        sar_source_dataset_index = src_ds_idx

        if self.mode == "normal":
            z_out = z
        elif self.mode == "zero":
            # Constant minimum-normalized SAR, not a physical "missing SAR"
            # token. After (VV+25)/25 and (VH+32.5)/32.5, 0.0 is the clip
            # floor (VV=-25 dB, VH=-32.5 dB). Shuffled remains the primary
            # geographic-correspondence test.
            z_out = torch.zeros_like(z)
        elif self.mode == "shuffled":
            sar_source_test_index = self.perm[i]
            sar_source_dataset_index = self.test_indices[sar_source_test_index]
            _, z_out, _ = self.root_dataset[sar_source_dataset_index]
        elif self.mode == "noise":
            z_out = noise_sar_like(z, self.noise_stats, self.noise_seed, i)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Slim collatable metadata only. Full sample_meta can contain None
        # (e.g. canonical_geographic_group) which breaks default_collate.
        meta = {
            "sample_id": sample_id_from_root(self.root_dataset, src_ds_idx),
            "test_index": int(i),
            "dataset_index": int(src_ds_idx),
            "sar_intervention": self.mode,
            "sar_source_test_index": int(sar_source_test_index),
            "sar_source_dataset_index": int(sar_source_dataset_index),
            "sar_source_sample_id": sample_id_from_root(
                self.root_dataset, sar_source_dataset_index
            ),
        }
        return y, z_out, x0, meta


def soft_region_l1(pred, target, weight, eps=1e-8):
    """Soft-weighted element-wise L1, same scale as ordinary L1.

    pred, target: [B, C, H, W]
    weight M:     [B, 1, H, W]

    Ordinary L1 is mean(|pred-target|) over all B*C*H*W elements.
    This is the same quantity with spatial/spectral weights:

        err = |pred - target|                         # [B,C,H,W]
        weights = M.expand_as(err)                    # [B,C,H,W]
        L1 = (weights * err).sum() / weights.sum()

    If M is identically 1, this equals torch.mean(|pred-target|).
    Expanding M over C makes the denominator count all 13 channels,
    so the result stays on the ordinary per-element L1 scale.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if weight.ndim != 4 or weight.shape[0] != pred.shape[0] or weight.shape[1] != 1:
        raise ValueError(
            f"weight must be [B,1,H,W], got {tuple(weight.shape)} for pred {tuple(pred.shape)}"
        )
    err = torch.abs(pred - target)
    weights = weight.expand_as(err)
    return (weights * err).sum() / weights.sum().clamp_min(eps)


def spectral_angle_map_deg(pred, target, eps=1e-8):
    """Per-pixel SAM in degrees.

    pred, target: [B, C, H, W]
    returns angle_map: [B, H, W]

    Matches src.utils.metrics.sam_deg's per-pixel angle; sam_deg then
    averages that map, while region metrics weight it by M[:,0].
    """
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"expected pred/target [B,C,H,W], got {tuple(pred.shape)} / {tuple(target.shape)}"
        )
    # [B,C,H,W] x [B,C,H,W] --sum_C--> [B,H,W]
    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.norm(pred, dim=1)
    target_norm = torch.norm(target, dim=1)
    cos = dot / (pred_norm * target_norm + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    angle_map = torch.acos(cos) * (180.0 / torch.pi)
    return angle_map


def soft_region_sam_deg(pred, target, weight, eps=1e-8):
    """Soft-weighted SAM in degrees using an explicit [B,H,W] mask.

    pred, target: [B, C, H, W]
    weight M:     [B, 1, H, W]
    angle_map:    [B, H, W]  (spectral angle at each pixel)
    m = M[:, 0]:  [B, H, W]

        cloud_sam = (m * angle_map).sum() / m.sum()
        clear_sam = ((1-m) * angle_map).sum() / (1-m).sum()
    """
    if weight.ndim != 4 or weight.shape[1] != 1:
        raise ValueError(f"weight must be [B,1,H,W], got {tuple(weight.shape)}")
    angle_map = spectral_angle_map_deg(pred, target, eps=eps)
    m = weight[:, 0]
    if angle_map.shape != m.shape:
        raise ValueError(
            f"angle_map {tuple(angle_map.shape)} vs M[:,0] {tuple(m.shape)}"
        )
    return (m * angle_map).sum() / m.sum().clamp_min(eps)


def write_per_sample_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_tables(root_dir: str, per_mode: dict, extras: Optional[dict] = None) -> dict:
    """Write summary.csv / summary.json with raw deltas vs normal."""
    metric_keys = [
        "L1",
        "PSNR",
        "SSIM",
        "SAM",
        "LPIPS",
        "FID",
        "cloud_L1_soft",
        "clear_L1_soft",
        "cloud_SAM_soft",
        "clear_SAM_soft",
    ]
    normal = per_mode.get("normal", {})
    rows = []
    delta = {}
    for mode in INTERVENTION_MODES:
        if mode not in per_mode:
            continue
        rec = dict(per_mode[mode])
        rec["sar_intervention"] = mode
        drow = {}
        if mode != "normal" and normal:
            for key in metric_keys:
                if key in rec and key in normal and rec[key] is not None and normal[key] is not None:
                    drow[key] = rec[key] - normal[key]
                    rec[f"delta_{key}"] = drow[key]
        if drow:
            delta[mode] = drow
        rows.append(rec)

    payload = {
        "baseline": "DBCR_MR_r3",
        "lower_is_better": list(LOWER_IS_BETTER),
        "higher_is_better": list(HIGHER_IS_BETTER),
        "delta_definition": "delta = intervention - normal (raw; sign not flipped)",
        "rows": {r["sar_intervention"]: r for r in rows},
        "delta_vs_normal": delta,
    }
    if extras:
        payload.update(extras)
    save_json(os.path.join(root_dir, "summary.json"), payload)

    fieldnames = ["sar_intervention"] + metric_keys
    for key in metric_keys:
        fieldnames.append(f"delta_{key}")
    csv_path = os.path.join(root_dir, "summary.csv")
    ensure_dir(root_dir)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in rows:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})
    return payload


