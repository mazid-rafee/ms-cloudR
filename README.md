# Cloud Removal — Modular DB-CR Pipeline

SEN12MS-CR cloud removal with a shared **DBCRNet** backbone and selectable
deterministic diffusion-bridge schedules.

## Progress so far

1. **DB-CR (original)** — sinusoidal bridge schedule.
2. **DBCR_MR_r3** — scalar mean-reverting bridge (`r=3`); same architecture/loss/sampler.
3. **DBCR_SpatialMR_r3** — spatially adaptive MR bridge using a **soft cloud score** map `M`; same architecture/loss; **NFE=1** inference unchanged.

All three are selectable in one codebase via `bridge_schedule` / config files.

**Environment:** use the `pylrt` conda env (`~/.conda/envs/pylrt`).

**Backup (pre-cleanup, 2026-09-08):**  
`../Cloud Removal_BACKUP_20260908/` (exploratory code archived there).

---

## Core layout

| Path | Role |
|------|------|
| `src/models/dbcr.py` | `DBCRNet` + `alpha_schedule` / `mean_reverting_alpha_schedule` / `get_alpha_schedule` |
| `src/utils/cloud_score.py` | Soft Sentinel-2 cloud score `M = compute_soft_cloud_score(y)` (detached) |
| `src/utils/spatial_bridge.py` | SpatialMR_r3: `r_map = r_max·M`, spatial `A(s)`, `x_t = x0 + A·(y−x0)` |
| `src/train.py` / `src/eval.py` | Train / eval (schedule-aware) |
| `src/datasets/sen12mscr_dataset.py` | Loader → `(y, z, x0)`; S2 bands `[B1…B12]` |
| `configs/dbcr.json` | Original |
| `configs/dbcr_mean_reverting.json` | Scalar MR_r3 |
| `configs/dbcr_spatial_mr_r3.json` | SpatialMR_r3 (`r_max=3`) |

---

## Quick start (`pylrt`)

```bash
# Original
python -m src.train --gpu 0 --config configs/dbcr.json --run_name DBCR_original

# Scalar mean-reverting (r=3)
python -m src.train --gpu 0 --config configs/dbcr_mean_reverting.json --run_name DBCR_MR_r3

# Spatial MR (soft cloud score → local rates; NFE=1)
python -m src.train --gpu 0 --config configs/dbcr_spatial_mr_r3.json \
  --run_name DBCR_SpatialMR_r3_seed42_50_epochs --seed 42 --epochs 50 --nfe 1
```

Background training example:
```bash
nohup python -m src.train --gpu 0 --config configs/dbcr_spatial_mr_r3.json \
  --run_name DBCR_SpatialMR_r3_seed42_50_epochs --seed 42 --epochs 50 --nfe 1 \
  > logs/DBCR_SpatialMR_r3_seed42_50_epochs.out 2>&1 &
```

Eval:
```bash
python -m src.eval --gpu 0 --config configs/dbcr_spatial_mr_r3.json \
  --checkpoint outputs/<run_name>/checkpoints/best.pt \
  --run_name <run_name>_eval --nfe 1 --bridge_schedule spatial_mr_r3
```

Region MAE by cloud score (after training):
```bash
python scripts/eval_cloud_region_metrics.py --gpu 0 \
  --checkpoint outputs/<run_name>/checkpoints/best.pt \
  --bridge_schedule spatial_mr_r3 --nfe 1 --seed 42 \
  --run_name <run_name>_region_eval
```

---

## Bridge schedules (modular)

| `bridge_schedule` | Config | Training `x_t` |
|-------------------|--------|----------------|
| `original` | `configs/dbcr.json` | `(1−α)x0 + α y`, `α=sin(π/2·t/T)` |
| `mean_reverting` / `mr_r3` | `configs/dbcr_mean_reverting.json` | same form, `α=(1−e^{−r s})/(1−e^{−r})`, `s=t/T`, `r=3` |
| `spatial_mr_r3` | `configs/dbcr_spatial_mr_r3.json` | `M=soft_cloud_score(y)`; `r_map=3·M`; spatial `A(s)`; `x_t = x0 + A·(y−x0)` |

**Unchanged across schedules:** `DBCRNet` (NAF + SAR `SFBlock`), L1 loss, Adam, data split/preprocessing, NFE=1 ODE inference (SpatialMR uses scalar MR alphas only for the one-step reverse).

### Soft cloud score (SpatialMR only)
- Input: normalized cloudy S2 `[B,13,H,W]` in `[0,1]` (no second `/10000`).
- Output: `M ∈ [B,1,H,W]` in `[0,1]`, **detached / non-trainable** (DSen2-CR-style spectral score + closing + blur).
- Diagnostic: Pearson(M, residual `|y−x0|`) ≈ **0.79** on train+val subsample.

### NFE note
At `NFE=1`, reverse sampling is schedule-independent (`α(T)=1`, `α(0)=0`). Differences vs original/MR mainly reflect the **training** bridge distribution.

### Architecture
Dual-stream U-Net: widths `(22,44,88,176)`, enc `(1,1,1,28)`, dec `(1,1,1,1)`, heads `(1,1,2,4)`, `time_dim=128` (~13.6M params).

### Preprocessing / metrics
- S2: clip `[0,10000]` → `/10000`; S1 VV/VH clipped and scaled to `[0,1]`.
- Test: L1, PSNR, SSIM, SAM(deg); optional LPIPS, FID.

---

## Recorded results (NFE=1)

Same split ≈97773 / 12221 / 12223, batch 4, lr `5e-5`, `T=1000` unless noted.

| Run | Schedule | Seed | Ep | L1 ↓ | PSNR ↑ | SSIM ↑ | SAM↓ |
|-----|----------|------|----|------|--------|--------|------|
| `DBCR_original_control_seed42_epochs50_20260812` | original | 42 | 50 | 0.011968 | 34.02 | 0.8868 | 3.63 |
| `DBCR_MR_r3_seed42_epochs50_20260812` | MR r=3 | 42 | 50 | **0.011096** | **34.51** | **0.8963** | **3.35** |
| `DBCR_original_control_seed123_epochs20_20260816` | original | 123 | 20 | 0.013784 | 32.94 | 0.8600 | 4.08 |
| `DBCR_MR_r3_seed123_epochs20_20260816` | MR r=3 | 123 | 20 | 0.013249 | 33.25 | 0.8684 | 3.94 |
| `DBCR_SpatialMR_r3_seed123_epochs20_20260909` | SpatialMR r=3 | 123 | 20 | *(running / see `test_metrics.json` when done)* |
| `DBCR_SpatialMR_r3_seed42_50_epochs` | SpatialMR r=3 | 42 | 50 | *(planned controlled ablation vs MR_r3 seed42)* |

**Finding so far:** scalar MR_r3 beats original at matched seed/epochs (NFE=1). SpatialMR is the next controlled ablation (soft-cloud local rates vs global `r=3`).

---

## Validation / diagnostic scripts

```bash
python scripts/check_bridge_schedules.py
python scripts/check_bridge_reverse_consistency.py
python scripts/validate_soft_cloud_score.py --gpu 0 --seasons winter
python scripts/test_spatial_mr_bridge.py
python scripts/validate_spatial_mr_bridge.py --gpu 0 --seasons winter
python scripts/analyze_cloud_score_vs_residual.py --gpu 0
```

---

## Lean keep-set

**Live repo:** DB-CR model/train/eval, cloud score + spatial bridge utils, three configs, helper scripts, `outputs/DBCR_*`, `invalid_files.txt`.

**Archived** (in backup): residual/scientific/SciBridge pipelines, reaction–transport / flow-action experiments, related tests/docs. See inventory in `Cloud Removal_BACKUP_20260908/` or older README history if needed.
