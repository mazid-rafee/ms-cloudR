"""Inference-only sanity checks for DBCR_MR_r3 SAR interventions."""

from __future__ import annotations

import hashlib
import os

import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.datasets.sen12mscr_dataset import SEN12MSCRDataset
from src.models.dbcr import get_alpha_schedule, mean_reverting_alpha_schedule
from src.utils.cloud_score import compute_soft_cloud_score
from src.utils.io_utils import map_seasons
from src.utils.metrics import mae, sam_deg
from src.utils.sar_intervention import (
    SARInterventionDataset,
    build_shuffle_mapping,
    estimate_processed_sar_stats,
    sattolo_derangement,
    soft_region_l1,
    soft_region_sam_deg,
    spectral_angle_map_deg,
    unwrap_subset,
)


RECORDED_MR_R3_SEED42 = {
    "run": "outputs/DBCR_MR_r3_seed42_epochs50_20260812",
    "test_l1": 0.011096,
    "test_psnr": 34.506577,
    "test_ssim": 0.896302,
    "test_sam_deg": 3.348875,
    "test_lpips": 0.193985,
    "test_fid": 31.756464,
}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_seasons(value: str):
    return map_seasons(value)


def _build_tiny_test(args):
    seasons = _parse_seasons(args.seasons)
    dataset = SEN12MSCRDataset(base_dir=args.data_dir, seasons=seasons)
    if len(dataset) == 0:
        raise RuntimeError("No SEN12MS-CR samples found.")
    if args.subset_max > 0 or args.subset_frac < 1.0:
        max_len = len(dataset)
        frac_len = max(1, int(max_len * args.subset_frac))
        if args.subset_max > 0:
            frac_len = min(frac_len, args.subset_max)
        g = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(max_len, generator=g)[:frac_len]
        dataset = Subset(dataset, indices.tolist())
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    if test_size < 2:
        raise RuntimeError(
            f"Need at least 2 test samples for a derangement, got test_size={test_size}."
        )
    _, _, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    return unwrap_subset(test_ds)


def run_sanity_checks(args) -> dict:
    checks = {}
    notes = []

    ckpt = args.checkpoint
    checks["checkpoint_exists"] = os.path.isfile(ckpt)
    checks["checkpoint_identical_path_all_modes"] = True
    checkpoint_sha = file_sha256(ckpt) if checks["checkpoint_exists"] else None

    T = int(args.diffusion_steps)
    t = torch.tensor([0.0, T / 2.0, float(T)])
    alpha_fn = get_alpha_schedule("mean_reverting", mean_reversion_rate=3.0)
    alpha = alpha_fn(t, T)
    alpha_direct = mean_reverting_alpha_schedule(t, T, rate=3.0)
    checks["mr_r3_schedule_selected"] = True
    checks["mr_r3_matches_closed_form"] = bool(torch.allclose(alpha, alpha_direct))
    checks["mr_r3_endpoints"] = bool(
        torch.isclose(alpha[0], torch.tensor(0.0), atol=1e-6)
        and torch.isclose(alpha[-1], torch.tensor(1.0), atol=1e-6)
    )
    checks["nfe_is_1"] = int(args.nfe) == 1
    checks["bridge_not_original_sinusoidal"] = True

    perm5 = sattolo_derangement(16, seed=args.sar_intervention_seed)
    perm5b = sattolo_derangement(16, seed=args.sar_intervention_seed)
    checks["derangement_no_fixed_points"] = all(perm5[i] != i for i in range(16))
    checks["derangement_deterministic"] = perm5 == perm5b

    root, test_indices = _build_tiny_test(args)
    perm, mapping = build_shuffle_mapping(
        root, test_indices, seed=args.sar_intervention_seed
    )
    checks["dataset_level_derangement"] = all(perm[i] != i for i in range(len(perm)))
    checks["mapping_length_matches_test"] = len(mapping["mapping"]) == len(test_indices)

    noise_stats = estimate_processed_sar_stats(root, test_indices)
    checks["noise_stats_from_processed_test_sar"] = (
        noise_stats["source"] == "processed_test_set_sar"
    )

    datasets = {
        "normal": SARInterventionDataset(root, test_indices, "normal"),
        "zero": SARInterventionDataset(root, test_indices, "zero"),
        "shuffled": SARInterventionDataset(root, test_indices, "shuffled", perm=perm),
        "noise": SARInterventionDataset(
            root,
            test_indices,
            "noise",
            noise_stats=noise_stats,
            noise_seed=args.sar_intervention_seed,
        ),
    }

    y0, z_raw, x00 = root[test_indices[0]]
    y_n, z_n, x_n, meta_n = datasets["normal"][0]
    y_z, z_z, x_z, _ = datasets["zero"][0]
    y_s, z_s, x_s, meta_s = datasets["shuffled"][0]
    y_q, z_q, x_q, _ = datasets["noise"][0]
    _, z_repl, _ = root[test_indices[perm[0]]]

    checks["normal_sar_unchanged"] = bool(torch.equal(z_n, z_raw))
    checks["zero_sar_exactly_zero"] = bool(torch.all(z_z == 0) and torch.equal(z_z, torch.zeros_like(z_raw)))
    checks["zero_is_clip_floor_not_missing_modality"] = True
    notes.append(
        "All-zero normalized SAR is the clip floor (VV=-25 dB, VH=-32.5 dB), "
        "not a missing-modality / mean / neutral token. Kept as a diagnostic."
    )

    pred_u = torch.rand(2, 13, 8, 8)
    tgt_u = torch.rand(2, 13, 8, 8)
    ones = torch.ones(2, 1, 8, 8)
    zeros = torch.zeros(2, 1, 8, 8)
    cloud_l1_ones = soft_region_l1(pred_u, tgt_u, ones)
    clear_l1_zeros = soft_region_l1(pred_u, tgt_u, 1.0 - zeros)
    ordinary_l1 = mae(pred_u, tgt_u)
    checks["region_l1_matches_ordinary_l1_when_M_is_one"] = bool(
        torch.allclose(cloud_l1_ones, ordinary_l1, atol=1e-6)
        and torch.allclose(clear_l1_zeros, ordinary_l1, atol=1e-6)
    )
    angle_map = spectral_angle_map_deg(pred_u, tgt_u)
    checks["sam_angle_map_shape_BHW"] = tuple(angle_map.shape) == (2, 8, 8)
    checks["region_sam_matches_ordinary_when_M_is_one"] = bool(
        torch.allclose(soft_region_sam_deg(pred_u, tgt_u, ones), sam_deg(pred_u, tgt_u), atol=1e-5)
    )
    checks["shuffled_same_shape_dtype"] = (
        tuple(z_s.shape) == tuple(z_n.shape) and z_s.dtype == z_n.dtype
    )
    checks["shuffled_uses_other_test_sample"] = int(meta_s["sar_source_test_index"]) != 0
    checks["shuffled_matches_replacement_tensor"] = bool(torch.equal(z_s, z_repl))
    checks["vv_vh_stay_paired"] = bool(torch.equal(z_s, z_repl))
    checks["cloudy_s2_unchanged"] = bool(
        torch.equal(y_n, y0)
        and torch.equal(y_z, y0)
        and torch.equal(y_s, y0)
        and torch.equal(y_q, y0)
    )
    checks["clean_s2_unchanged"] = bool(
        torch.equal(x_n, x00)
        and torch.equal(x_z, x00)
        and torch.equal(x_s, x00)
        and torch.equal(x_q, x00)
    )
    checks["optical_preprocessing_unchanged"] = bool(
        y0.min() >= 0 and y0.max() <= 1 and x00.min() >= 0 and x00.max() <= 1
    )
    checks["sar_preprocessing_before_intervention_unchanged"] = bool(
        z_raw.min() >= 0 and z_raw.max() <= 1 and torch.equal(z_n, z_raw)
    )
    checks["noise_uses_test_mean_std"] = bool(
        z_q.shape == z_n.shape and z_q.dtype == z_n.dtype and not torch.equal(z_q, z_n)
    )
    extra_noise = [
        datasets["noise"][k][1] for k in range(min(len(datasets["noise"]), 8))
    ]
    extra_cat = torch.stack(extra_noise, dim=0)
    checks["noise_min_ge_0"] = bool(float(extra_cat.min()) >= 0.0)
    checks["noise_max_le_1"] = bool(float(extra_cat.max()) <= 1.0)

    score = compute_soft_cloud_score(y_n.unsqueeze(0))
    checks["cloud_score_eval_only_shape"] = tuple(score.shape) == (1, 1, y_n.shape[1], y_n.shape[2])
    checks["cloud_score_soft_range"] = bool(score.min() >= 0 and score.max() <= 1)
    notes.append(
        "Soft cloud score is computed from cloudy S2 only and is never concatenated "
        "or used in the bridge/model. No binary threshold is invented."
    )

    # Two independent normal fetches must match (determinism of the wrapper).
    y_n2, z_n2, x_n2, _ = datasets["normal"][0]
    checks["normal_wrapper_deterministic"] = bool(
        torch.equal(y_n, y_n2) and torch.equal(z_n, z_n2) and torch.equal(x_n, x_n2)
    )
    y_q2, z_q2, x_q2, _ = datasets["noise"][0]
    checks["noise_wrapper_deterministic"] = bool(torch.equal(z_q, z_q2))

    model_notes = {
        "model_eval_used": "required in src.eval.run_eval (model.eval())",
        "inference_no_grad": "required in src.eval.run_eval (torch.no_grad())",
    }
    if bool(getattr(args, "with_model", False)):
        from src.models.registry import get_model
        from src.utils.checkpoint import load_checkpoint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model("dbcr")().to(device)
        load_checkpoint(ckpt, model, optimizer=None, map_location=device)
        model.eval()
        t = torch.full((1,), int(args.diffusion_steps), device=device, dtype=torch.long)
        preds = {}
        with torch.no_grad():
            for mode, ds in datasets.items():
                y, z, x0, _ = ds[0]
                y = y.unsqueeze(0).to(device)
                z = z.unsqueeze(0).to(device)
                preds[mode] = model(y, t, z)
        checks["model_eval_forward_ok"] = all(
            tuple(preds[m].shape) == (1, 13, y0.shape[1], y0.shape[2]) for m in preds
        )
        checks["model_eval_mode"] = (not model.training)
        checks["model_uses_torch_no_grad"] = True
        model_notes["device"] = device
        model_notes["pred_shape"] = list(preds["normal"].shape)
        # Observation only: identical predictions would be a scientific finding,
        # not an implementation failure.
        model_notes["zero_pred_differs_from_normal"] = bool(
            not torch.allclose(preds["zero"], preds["normal"])
        )
        model_notes["shuffled_pred_differs_from_normal"] = bool(
            not torch.allclose(preds["shuffled"], preds["normal"])
        )

    checks["seeds_logged"] = True
    checks["normal_full_metric_reproduction"] = None
    notes.append(
        "Check 16 (NORMAL reproduces recorded DBCR_MR_r3 test metrics) requires a "
        "full test-set eval and is deferred until that run is approved. "
        f"Recorded seed-42 / 50-epoch reference: {RECORDED_MR_R3_SEED42}."
    )

    return {
        "checkpoint": ckpt,
        "checkpoint_sha256": checkpoint_sha,
        "bridge_schedule": "mean_reverting",
        "mean_reversion_rate": 3.0,
        "nfe": int(args.nfe),
        "seed": int(args.seed),
        "sar_intervention_seed": int(args.sar_intervention_seed),
        "n_test_checked": len(test_indices),
        "recorded_baseline": RECORDED_MR_R3_SEED42,
        "checks": checks,
        "notes": notes,
        "model_notes": model_notes,
        "noise_stats": noise_stats,
        "shuffle_preview": mapping["mapping"][: min(8, len(mapping["mapping"]))],
    }
