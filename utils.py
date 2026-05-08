"""
utils.py — Shared utilities: focal loss, metrics, seeding, etc.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(cfg: dict, device: torch.device) -> nn.Module:
    class_weights = torch.tensor(cfg["classes"]["weights"], dtype=torch.float32).to(device)

    loss_type = cfg["training"].get("loss", "focal")
    if loss_type == "focal":
        return FocalLoss(
            gamma=cfg["training"].get("focal_gamma", 2.0),
            weight=class_weights,
        )
    else:
        return nn.CrossEntropyLoss(weight=class_weights)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["BIRD", "DRONE", "IRRELEVANT"]


def compute_macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def print_metrics(y_true, y_pred, split: str = "val"):
    macro_f1 = compute_macro_f1(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {split.upper()} — Macro F1: {macro_f1:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"{'':12s}", "  ".join(f"{n:>10s}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:12s}", "  ".join(f"{v:10d}" for v in row))
    print()
    return macro_f1


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    total_epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 5)
    sched_type = cfg["training"].get("scheduler", "cosine")

    if sched_type == "cosine":
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg["training"]["lr"],
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / total_epochs,
            anneal_strategy="cos",
        )
    else:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    return scheduler


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    best_f1 = ckpt.get("best_f1", 0.0)
    print(f"  ✓ Loaded checkpoint from epoch {epoch}, best F1={best_f1:.4f}")
    return epoch, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Frame differencing augmentation (extra motion signal)
# ──────────────────────────────────────────────────────────────────────────────

def add_frame_diff(images: torch.Tensor) -> torch.Tensor:
    """
    images: (B, T, C, H, W)
    Returns: (B, T, C+C, H, W) by appending frame[t] - frame[t-1]
    """
    B, T, C, H, W = images.shape
    diff = torch.zeros_like(images)
    diff[:, 1:] = images[:, 1:] - images[:, :-1]
    return torch.cat([images, diff], dim=2)  # (B, T, 2C, H, W)
