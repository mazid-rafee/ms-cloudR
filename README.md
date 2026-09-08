# Cloud Removal — Lean DB-CR Pipeline

## Real progress so far

1. **DB-CR baseline** — deterministic diffusion bridge for SEN12MS-CR cloud removal (13-band optical + 2-channel SAR).
2. **Mean-reverting DB-CR ablation** — same `DBCRNet` / same loss / same sampler; only the bridge schedule `α(t)` changed to a deterministic mean-reverting trajectory (`rate=3.0`).

That is the validated core of this repo. Everything else below is exploratory work documented so it can be restored from backup later if useful.

**Pre-cleanup backup (created 2026-09-08):**  
`/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Removal_BACKUP_20260908/`  
(~1.9G code/outputs; `data/` excluded and symlinked; see `BACKUP_INFO.txt`.)

**Cleanup status (2026-09-08):** exploratory code/outputs removed from the live repo per your approval. Inventory below remains as a map of what still exists in the backup.

---

## Core structure

| Path | Role |
|------|------|
| `src/models/dbcr.py` | `DBCRNet`, `alpha_schedule`, `mean_reverting_alpha_schedule`, `get_alpha_schedule` |
| `src/train.py` / `src/eval.py` | DB-CR train / eval |
| `src/datasets/sen12mscr_dataset.py` | SEN12MS-CR loader → `(y, z, x0)` |
| `src/datasets/sen12mscr_utils.py` | Season/path helpers used by the loader |
| `src/utils/` | config, checkpoint, metrics (L1/PSNR/SSIM/SAM), logging, I/O |
| `configs/dbcr.json` | Original sinusoidal bridge |
| `configs/dbcr_mean_reverting.json` | Mean-reverting bridge (`rate=3.0`) |
| `scripts/check_bridge_schedules.py` | Prints α at `t/T ∈ {0,0.25,0.5,0.75,1}` |
| `scripts/check_bridge_reverse_consistency.py` | Oracle reverse-bridge consistency (NFE 1/3/5) |
| `scripts/scan_invalid.py` | Invalid-triplet scanner → `outputs/invalid_files.txt` |
| `data/SEN12MS-CR/` | Dataset (~381G; not duplicated in backup) |
| `outputs/DBCR_*` | DB-CR / MR run artifacts |

---

## Quick Start

Original DB-CR:
```bash
python3 -m src.train --gpu 0 --config configs/dbcr.json --run_name DBCR_original
```

Mean-reverting DB-CR (`rate=3.0`):
```bash
python3 -m src.train --gpu 0 --config configs/dbcr_mean_reverting.json --run_name DBCR_MR_r3
```

Eval:
```bash
python3 -m src.eval --gpu 0 --config configs/dbcr.json \
  --checkpoint outputs/<run_name>/checkpoints/best.pt
```

Sanity checks (no training):
```bash
python3 scripts/check_bridge_schedules.py
python3 scripts/check_bridge_reverse_consistency.py
```

---

## DB-CR method (core)

### Bridge schedules

| Schedule | Formula | Config flag |
|----------|---------|-------------|
| `original` | `α(t) = sin(π/2 · t/T)` | `bridge_schedule=original` |
| `mean_reverting` | `α(t) = (1−e^{−r s})/(1−e^{−r})`, `s=t/T`, default `r=3` | `bridge_schedule=mean_reverting` |

Bridge state (same for both):
```
x_t = (1 − α_t) · x0 + α_t · y
```
- `x0`: clean S2 `[B,13,H,W]`
- `y`: cloudy S2 `[B,13,H,W]`
- `z`: SAR VV/VH `[B,2,H,W]` (conditioning only; not in the bridge mix)

Network predicts `x0` directly; train with **global L1**.  
Deterministic reverse ODE (no noise):
```
x_next = (1 − α_next/α_curr)·x̂0 + (α_next/α_curr)·x_curr
```
Default `NFE=1` (timesteps `[T, 0]`). Optional `NFE=3` / `5`.

**Interpretation:** at `NFE=1`, reverse sampling is schedule-independent because `α(T)=1` and `α(0)=0`. Original vs MR gaps at `NFE=1` mainly measure the **training** bridge distribution. At `NFE=3/5`, training distribution **and** reverse trajectory both matter.

### Architecture (unchanged across schedules)
- Dual-stream U-Net: optical + SAR encoders (`NAFBlock`), `SFBlock` fusion (opt queries SAR), optical decoder
- Widths `(22,44,88,176)`, enc blocks `(1,1,1,28)`, dec `(1,1,1,1)`, heads `(1,1,2,4)`, `time_dim=128`
- ~13.6M trainable parameters

### Preprocessing
- S2: clip `[0,10000]` → `/10000`
- S1: VV `[-25,0]`, VH `[-32.5,0]` → shift/scale to `[0,1]`
- No spatial augmentation

### Metrics
L1/MAE, PSNR, SSIM, SAM(deg); optional LPIPS/FID if installed.

---

## DB-CR experimental results (recorded)

All below used the same split sizes (≈97773 / 12221 / 12223), batch 4, lr `5e-5`, `T=1000`, `NFE=1` at end-of-train test unless noted.

| Run folder | Schedule | Seed | Epochs | L1 ↓ | PSNR ↑ | SSIM ↑ | SAM↓ | LPIPS | FID |
|------------|----------|------|--------|------|--------|--------|------|-------|-----|
| `DBCR_20260209` | original (pre-flag) | 42 | 50 | 0.011318 | 34.37 | 0.8939 | 3.42 | 0.195 | 31.9 |
| `DBCR_20260707` | original (pre-flag) | 42 | 50 | 0.011522 | 34.23 | 0.8903 | 3.47 | 0.198 | 33.2 |
| `DBCR_original_control_seed42_epochs50_20260812` | original | 42 | 50 | 0.011968 | 34.02 | 0.8868 | 3.63 | 0.202 | 32.9 |
| `DBCR_MR_r3_seed42_epochs50_20260812` | mean_reverting r=3 | 42 | 50 | **0.011096** | **34.51** | **0.8963** | **3.35** | **0.194** | **31.8** |
| `DBCR_original_control_seed123_epochs20_20260816` | original | 123 | 20 | 0.013784 | 32.94 | 0.8600 | 4.08 | 0.211 | 39.7 |
| `DBCR_MR_r3_seed123_epochs20_20260816` | mean_reverting r=3 | 123 | 20 | 0.013249 | 33.25 | 0.8684 | 3.94 | 0.211 | 38.0 |

**Finding (controlled ablation):** with matched seed/epochs, mean-reverting `r=3` beat original on L1/PSNR/SSIM/SAM for both the 50-epoch (seed 42) and 20-epoch (seed 123) pairs at default `NFE=1`. Treat this as a training-distribution effect until multi-NFE eval is completed.

**Math checks already done:**
- Endpoint properties `α(0)=0`, `α(T)=1` for both schedules
- Oracle reverse consistency passed for NFE ∈ {1,3,5} on both schedules
- `NFE=1` schedule independence confirmed analytically/numerically

---

# Inventory of exploratory findings (removed from live repo; restore from backup)

> These are **not** part of the lean DB-CR core. Listed with enough detail to decide later whether to revive them.

## A. Residual / scientific CR pipelines (`src/`)

### A1. Residual CR (`src/models/residual_cr/`, `train_residual_cr*.py`)
- **Idea:** shallow residual decoder: `x̂ = clamp(y + δ, 0, 1)` — no diffusion/PDE/GAN.
- **Inputs:** cloudy S2 ± optional SAR.
- **Variants:**
  - `train_residual_cr.py` — basic residual baseline
  - `train_residual_cr_band_balanced.py` — band-balanced L1
  - `train_residual_cr_all_seasons.py` — all-season residual training
- **Outputs:** `outputs/sen12mscr_residual_baseline`, `sen12mscr_residual_band_balanced`, `sen12mscr_residual_converged`, `sen12mscr_all_seasons_original`, `sen12mscr_all_seasons_balanced_raw`
- **Why keep in archive:** simple non-bridge baseline for comparing against DB-CR; band-balancing for rare bands (e.g. B10).

### A2. Gated SAR residual (`residual_cr_gated_sar.py`, `train_residual_cr_gated_sar.py`, `evaluate_gated_sar_scientific.py`)
- **Idea:** spatially varying per-band SAR gate in `(0,1)` (sigmoid); study when SAR helps vs hurts.
- **Eval suite:** paired vs shuffled SAR, per-band MAE, spectral indices (NDVI/NDWI/NBR), gate analysis.
- **Outputs:** `outputs/sen12mscr_all_seasons_gated_sar`
- **Why archive:** SAR reliability / intervention analysis beyond DB-CR’s fixed SFBlock fusion.

### A3. Scientific CR (`src/models/scientific_cr/`, `train_scientific_cr.py`, `eval_scientific_cr.py`)
- **Idea:** optical encoder + SAR encoder + **ReliabilityGateFusion** + decoder; optional **uncertainty** (log-variance) head.
- **Config:** `configs/scientific_cr_baseline.json`
- **Outputs:** `outputs/scientific_cr_baseline`
- **Why archive:** early uncertainty / reliability-gated fusion design (related to later SciBridge ideas).

### A4. SciBridge-CR residual flow (`src/models/scibridge_cr/`, `train_scibridge_cr.py`)
- **Idea:** “bias-aware conditional residual flow matching”:
  - frozen residual **anchor**
  - **BiasCorrectionHead**
  - **ConditioningEncoder** (cacheable inference context)
  - **ConditionalResidualFlow** + ODE/Heun `integrate_flow` (`flow_solver.py`)
  - NFE ablation / parameter budget profiling
- **Outputs:** `outputs/scibridge_cr_residual_flow_poc`
- **Why archive:** closest non-DBCR “bridge/flow” research line; reusable conditioning/bias/flow pieces.

### A5. Losses & scientific metrics
| Path | Content |
|------|---------|
| `src/losses/residual_cr.py` | Ordinary L1, per-band L1, SAM loss wrappers |
| `src/losses/scientific_fidelity.py` | Spectral angle loss, spectral indices helpers |
| `src/metrics/band_statistics.py` | Train-split band stats / inverse-scale weights |
| `src/metrics/region_fidelity.py` | Spectral-corruption **proxy regions** (eval-only; not train input) |
| `src/metrics/residual_cr_metrics.py` | Residual baseline metrics + Pearson etc. |
| `src/metrics/scientific_metrics.py` | Fidelity + uncertainty coverage levels |

### A6. Dataset audits / diagnostics
| Path | Content |
|------|---------|
| `src/audit_sen12mscr_seasons.py` | Season audit |
| `src/compile_sen12mscr_report.py` | Report compilation |
| `src/diagnose_b10_clamp.py` | B10 clamp diagnostic |
| `src/datasets/sen12mscr_utils.py` | Extra dataset helpers for residual pipelines |
| `outputs/sen12mscr_season_audit`, `sen12mscr_b10_clamp_diagnostic`, `sen12mscr_analysis_bundle` | Artifacts |

---

## B. Physics / flow-action experiments (`experiments/`)

These are mostly **synthetic / theoretical** studies about reaction–transport and flow actions — not the production DB-CR train loop.

### B1. `reaction_transport_oracle/`
- Analytic fields/trajectories on a 2D periodic torus.
- Differentiable periodic finite-volume RT operators; mass / reaction-balance diagnostics.
- **Finding role:** ground-truth oracle for PDE bridge ideas before learning.
- Outputs: `outputs/reaction_transport_oracle`

### B2. `reaction_transport_learning/`
- Supervised recovery of velocity `v` and reaction `g` from synthetic RT trajectories.
- Modes: transport-only / reaction-only / combined; dynamics + smoothness + action losses.
- Diagnostics for failure modes independent of full training.
- Outputs: `outputs/reaction_transport_learning`

### B3. `material_coordinate_rt/`
- Material-coordinate constrained RT / Jacobian-corrected **flow-map identity**.
- Strong validation gate: discrete oracle must satisfy material identity before implementing correlation/soft-argmax estimator (model/loss stages intentionally blocked until gate passes).
- Outputs: `outputs/material_coordinate_rt`

### B4. `flow_action_identifiability/`
- Static **unbalanced transport/reaction action**; displacement-energy landscapes.
- Identifiability across normalizations (`normalized_rho0`, `rho0`, geometric, unweighted) and κ sweeps.
- Outputs: `outputs/flow_action_identifiability`

### B5. `structured_flow_action/`
- Structured unbalanced action: L2 + Charbonnier TV + **spectral low-rank** prior.
- Observability-aware structured actions.
- Outputs: `outputs/structured_flow_action`

### B6. `flow_action_robustness/`
- Strict robustness/generalization gate: subpixel offsets, resolution sweep, noise, TV retune, reaction families.
- Frozen observability threshold experiments.
- Outputs: `outputs/flow_action_robustness`

### B7. `sar_anchored_flow_action/`
- SAR-anchored optical RT action using **structure tensors** (optical vs SAR alignment cost).
- Extends structured action with `λ_sar` term.
- Outputs: `outputs/sar_anchored_flow_action`

### B8. `learned_cross_modal_flow/`
- Learned optical↔SAR structural correspondence for **global flow**.
- Correlation volume + soft-argmax + quadratic subpixel refinement; contrastive + confidence losses; speckle-strength robustness; bootstrap CIs.
- Synthetic balanced dataset; labels never fed to encoders.
- Outputs: `outputs/learned_cross_modal_flow`

### B9. Design docs (`docs/`)
| Doc | Content |
|-----|---------|
| `docs/reaction_transport_bridge_derivation.md` | Design-only derivation to replace DB-CR linear path with learned RT PDE fields `(v,g)`; **explicitly out of scope for modifying DBCRNet/train** at write time |
| `docs/reaction_transport_flow_map.md` | Flow-map / material-coordinate notes tied to the RT line |

### B10. Tests (`tests/`)
All current tests target exploratory lines (residual CR, scientific CR, RT oracle/learning, material RT, flow-action*, cross-modal flow). **No dedicated DB-CR unit tests yet** (schedule checks live under `scripts/`).

---

## C. Conceptual map: how extras relate to DB-CR

```
DB-CR core (KEEP)
  ├── sinusoidal α(t)          ← original
  └── mean-reverting α(t)      ← ablation r=3  [DONE]

Explored alternatives (ARCHIVE)
  ├── Residual CR / band-balanced / gated SAR   → non-bridge regression baselines
  ├── Scientific CR + uncertainty               → reliability / aleatoric heads
  ├── SciBridge residual flow                   → flow-matching + bias correction
  └── Reaction–transport / flow-action suite    → PDE / action theory for a future
                                                  dynamical bridge (docs + synthetic
                                                  experiments; not wired into train.py)
```

**Mean-reverting schedule** is the only completed one-variable change to the DB-CR bridge. RT/flow-action work is the longer-horizon idea for replacing the *path prescription* itself — documented, not integrated.

---

## Lean keep-set (cleanup applied)

**Kept in live repo:**
- `src/models/dbcr.py`, `registry.py`
- `src/train.py`, `src/eval.py`, `src/main.py`, `src/__init__.py`
- `src/datasets/sen12mscr_dataset.py`, `sen12mscr_utils.py`
- `src/utils/` (`checkpoint`, `config`, `metrics`, `logger`, `io_utils`, `image_utils`)
- `configs/dbcr.json`, `configs/dbcr_mean_reverting.json`
- Scripts: `check_bridge_*.py`, `scan_invalid.py`, `analyze_invalid.py`, `gpu_manager.py`, `model_stats.py`, `helper_scripts.txt`
- `data/`, `.gitignore`, this `README.md`
- `outputs/DBCR_*`, `outputs/invalid_files.txt`

**Removed from live repo (still in backup):**
`experiments/`, `tests/`, exploratory `docs/`, residual/scientific/SciBridge models & trains/evals, `src/losses/`, `src/metrics/`, non-DBCR configs, non-`DBCR_*` outputs, unused `src/utils/diffusion.py`.

Restore any item from:
`/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Removal_BACKUP_20260908/`
