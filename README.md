# Cloud Removal - Standardized ML Pipeline

This repo provides a standardized training/evaluation pipeline for cloud removal
on SEN12MS-CR data using the DB-CR diffusion bridge model.

## Structure
- `src/datasets/`: data loading
- `src/models/`: model definitions
- `src/utils/`: logging, checkpoints, metrics
- `src/train.py`: DB-CR training + validation + testing
- `src/eval.py`: standalone DB-CR evaluation
- `src/models/registry.py`: model registry (DB-CR now; others stubbed)
- `configs/`: run configs (JSON)
- `scripts/`: helper utilities (e.g., GPU memory manager)
- `outputs/`: saved runs, logs, metrics, checkpoints

## Quick Start
Train (DB-CR defaults: 50 epochs, batch size 4, lr 5e-5):
```
python -m src.train --gpu 0
```

Evaluate a checkpoint (ODE inference, NFE=1 default):
```
python -m src.eval --gpu 0 --checkpoint outputs/<run_name>/checkpoints/best.pt
```

Use a config file:
```
python -m src.train --gpu 0 --config configs/dbcr.json
python -m src.eval --gpu 0 --config configs/dbcr.json --checkpoint outputs/<run_name>/checkpoints/best.pt
```

Or use the dispatcher:
```
python -m src.main --mode train --gpu 0
python -m src.main --mode eval --gpu 0 --checkpoint outputs/<run_name>/checkpoints/best.pt
```

## DB-CR specifics
- Diffusion bridge with deterministic inference (no stochastic noise).
- Bridge state: `x_t = (1 - α_t) * x0 + α_t * y`, `α_t = sin(π/2 * t/T)`.
- Network predicts `x0` directly and is trained with L1 loss.
- ODE-like inference update:
  `x_{t-s} = (1 - α_{t-s}/α_t) * x̂0 + (α_{t-s}/α_t) * x_t`.
- Default `NFE=1`, optional `--nfe 3` or `--nfe 5`.
- Official split sizes: train 114,056; val 7,176; test 7,899 (falls back to 80/10/10 if local dataset is smaller).
- Preprocessing:
  - S2 (13 bands): clip [0,10000] then scale to [0,1].
  - S1 (VV,VH): clip VV [-25,0], VH [-32.5,0], shift to positive then scale to [0,1].

## Metrics and images
- Test/Eval metrics: L1, PSNR, SSIM (saved in `test_metrics.json` / `eval_metrics.json`).
- To save sample outputs:
```
python -m src.train --save_images --num_save_images 4
python -m src.eval --save_images --num_save_images 4 --checkpoint outputs/<run_name>/checkpoints/best.pt
```
- Saved under `outputs/<run_name>/images/` as `.npy` (all bands) and `.png` (RGB).

GPU memory manager:
```
python scripts/gpu_manager.py --alloc 15 --gpu 0
python scripts/gpu_manager.py --free --gpu 0
```
PID files are stored next to the script in `scripts/`.

## Outputs
Each run writes to `outputs/<run_name>/`:
- `config.json`
- `metrics.csv`
- `test_metrics.json`
- `checkpoints/`
- `logs/`
`outputs/` contains generated artifacts and is typically not committed.