"""Training-only SAR semantic-alignment helpers for DBCR_MR_r3_SARAlign.

Design notes
------------
- A frozen DBCR_MR_r3 teacher provides clean-optical stage-3 anchors (Fc).
- The student predicts x0_hat from (x_t, t, z) exactly as baseline DBCR.
- L_align compares student pre-fusion sar_enc[3] (Fs) to detached Fc.
- Gradients from L_align update student SAR-only params AND shared downs;
  they do NOT update the frozen teacher. Optical-only student params are
  updated only by L_recon (and not by L_align, except indirectly later via
  shared downs already updated by L_align).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dbcr import DBCRNet
from src.utils.checkpoint import load_checkpoint


DEFAULT_TEACHER_CHECKPOINT = (
    "outputs/DBCR_MR_r3_seed42_epochs50_20260812/checkpoints/best.pt"
)


def spatial_cosine_align_loss(
    fs: torch.Tensor,
    fc: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-spatial-location cosine distance: mean(1 - cos(Fs, Fc)).

    Args:
        fs: student SAR feature [B, C, H, W], requires grad.
        fc: detached teacher optical feature [B, C, H, W].
    """
    if fs.shape != fc.shape:
        raise ValueError(f"Fs/Fc shape mismatch: {tuple(fs.shape)} vs {tuple(fc.shape)}")
    # cosine_similarity over channels -> [B, H, W]
    cos = F.cosine_similarity(fs, fc, dim=1, eps=eps)
    return torch.mean(1.0 - cos)


def feature_representation_stats(
    fs: torch.Tensor,
    fc: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Detach-safe Fs/Fc diagnostics for SARAlign logging."""
    with torch.no_grad():
        fs_d = fs.detach()
        fc_d = fc.detach()
        cos = F.cosine_similarity(fs_d, fc_d, dim=1, eps=eps)
        fs_l2 = torch.linalg.vector_norm(fs_d, ord=2, dim=1)
        fc_l2 = torch.linalg.vector_norm(fc_d, ord=2, dim=1)
        # Spatial std over H,W for each (B,C), then average.
        fs_spatial_std = fs_d.std(dim=(2, 3), unbiased=False).mean()
        finite = bool(
            torch.isfinite(fs_d).all()
            and torch.isfinite(fc_d).all()
            and torch.isfinite(cos).all()
        )
        return {
            "mean_cosine_similarity": float(cos.mean().item()),
            "mean_fs_l2": float(fs_l2.mean().item()),
            "mean_fc_l2": float(fc_l2.mean().item()),
            "fs_spatial_std": float(fs_spatial_std.item()),
            "features_finite": finite,
        }


def state_dict_checksum(module: nn.Module) -> str:
    """Stable checksum of all parameter tensors (for freeze/change audits)."""
    import hashlib

    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def named_param_checksum(module: nn.Module, prefixes: Tuple[str, ...]) -> str:
    import hashlib

    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        if not any(name.startswith(pref) for pref in prefixes):
            continue
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_module(module: nn.Module) -> nn.Module:
    """Eval mode + requires_grad=False for every parameter."""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


def load_frozen_teacher(
    checkpoint: str,
    device: torch.device | str,
    map_location=None,
) -> DBCRNet:
    """Load a full DBCR_MR_r3 checkpoint as a frozen training-only teacher.

    The full architecture is loaded so weights match the controlled baseline,
    but training only calls extract_optical_stage3 (optical path through stage 3).
    """
    path = str(checkpoint)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SARAlign teacher checkpoint not found: {path}")
    teacher = DBCRNet()
    load_checkpoint(
        path,
        teacher,
        optimizer=None,
        map_location=map_location if map_location is not None else device,
    )
    teacher.to(device)
    freeze_module(teacher)
    return teacher


@torch.no_grad()
def extract_detached_optical_anchor(
    teacher: DBCRNet,
    x0: torch.Tensor,
    anchor_t: float = 0.0,
) -> torch.Tensor:
    """Fc = stop_gradient(teacher optical stage-3 on clean x0 at fixed t)."""
    if teacher.training:
        teacher.eval()
    b = x0.size(0)
    t = torch.full((b,), float(anchor_t), device=x0.device, dtype=torch.float32)
    fc = teacher.extract_optical_stage3(x0, t)
    return fc.detach()


def teacher_requires_grad_any(teacher: nn.Module) -> bool:
    return any(p.requires_grad for p in teacher.parameters())


def parameter_group_tensors(model: DBCRNet) -> Dict[str, List[nn.Parameter]]:
    """Partition student params for gradient-norm audits."""
    groups = {
        "sar_only": [],
        "shared_downs": [],
        "optical_student": [],
        "other": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("sar_stem") or name.startswith("sar_enc"):
            groups["sar_only"].append(p)
        elif name.startswith("downs"):
            groups["shared_downs"].append(p)
        elif (
            name.startswith("opt_stem")
            or name.startswith("opt_enc")
            or name.startswith("opt_dec")
            or name.startswith("time_")
            or name.startswith("fuse")
            or name.startswith("mid")
            or name.startswith("ups")
            or name.startswith("head")
        ):
            groups["optical_student"].append(p)
        else:
            groups["other"].append(p)
    return groups


def grad_norm(params: Iterable[nn.Parameter]) -> float:
    total = 0.0
    found = False
    for p in params:
        if p.grad is None:
            continue
        found = True
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    if not found:
        return 0.0
    return total ** 0.5


def count_params_with_grad(module: nn.Module) -> Tuple[int, int]:
    """Return (n_params_requiring_grad, n_params_total)."""
    n_req = 0
    n_tot = 0
    for p in module.parameters():
        n = p.numel()
        n_tot += n
        if p.requires_grad:
            n_req += n
    return n_req, n_tot


def assert_teacher_frozen(teacher: nn.Module) -> None:
    if teacher.training:
        raise AssertionError("Teacher must remain in eval() mode")
    if teacher_requires_grad_any(teacher):
        raise AssertionError("Teacher has parameters with requires_grad=True")


def optimizer_contains_params(
    optimizer: torch.optim.Optimizer,
    params: Iterable[nn.Parameter],
) -> bool:
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    return any(id(p) in opt_ids for p in params)
