# Cloud Removal - Standardized ML Pipeline

This repo provides a standardized training/evaluation pipeline for cloud removal
on SEN12MS-CR data using a diffusion-style U-Net.

## Structure
- `src/datasets/`: data loading
- `src/models/`: model definitions
- `src/utils/`: logging, checkpoints, metrics
- `src/train.py`: training + validation + testing
- `src/eval.py`: standalone evaluation
- `scripts/`: helper utilities (e.g., GPU memory manager)
- `outputs/`: saved runs, logs, metrics, checkpoints

## Quick Start
Train:
```
python -m src.train --gpu 0
```

Evaluate a checkpoint:
```
python -m src.eval --gpu 0 --checkpoint outputs/<run_name>/checkpoints/best.pt
```

Or use the dispatcher:
```
python -m src.main --mode train --gpu 0
python -m src.main --mode eval --gpu 0 --checkpoint outputs/<run_name>/checkpoints/best.pt
```

GPU memory manager:
```
python scripts/gpu_manager.py --alloc 15 --gpu 0
python scripts/gpu_manager.py --free --gpu 0
```

## Outputs
Each run writes to `outputs/<run_name>/`:
- `config.json`
- `metrics.csv`
- `test_metrics.json`
- `checkpoints/`
- `logs/`
