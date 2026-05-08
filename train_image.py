"""
train_image.py — Phase 2: Train CNN + temporal image-sequence model.

Usage:
    python -m src.train_image
    python -m src.train_image --resume checkpoints/image_best.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VisgridDataset
from src.models.image_model import ImageSequenceModel
from src.utils import (
    set_seed, load_config, build_loss, build_scheduler,
    save_checkpoint, load_checkpoint, compute_macro_f1, print_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--data_root", default=None)
    p.add_argument("--resume",    default=None, help="Path to checkpoint to resume from")
    p.add_argument("--epochs",    type=int, default=None)
    return p.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    avg_loss = total_loss / len(y_true)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    return avg_loss, macro_f1, y_true, y_pred


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root      = Path(args.data_root or cfg["data"]["root"])
    train_dir = root / cfg["data"]["train_dir"]
    val_dir   = root / cfg["data"]["val_dir"]
    epochs    = args.epochs or cfg["training"]["epochs"]

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = VisgridDataset(
        str(train_dir),
        max_frames=cfg["data"]["max_frames"],
        max_traj_len=cfg["data"]["max_traj_len"],
    )
    val_ds = VisgridDataset(
        str(val_dir),
        max_frames=cfg["data"]["max_frames"],
        max_traj_len=cfg["data"]["max_traj_len"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    print(f"Train: {len(train_ds)} objects | Val: {len(val_ds)} objects")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ImageSequenceModel(
        backbone=cfg["image_model"]["backbone"],
        pretrained=cfg["image_model"]["pretrained"],
        embed_dim=cfg["image_model"]["embed_dim"],
        temporal=cfg["image_model"]["temporal"],
        num_temporal_layers=cfg["image_model"]["num_temporal_layers"],
        dropout=cfg["image_model"]["dropout"],
        max_frames=cfg["data"]["max_frames"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = build_loss(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    start_epoch, best_f1 = 0, 0.0
    ckpt_dir = Path(cfg["paths"]["checkpoints"])

    if args.resume:
        start_epoch, best_f1 = load_checkpoint(args.resume, model, optimizer, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs …\n")
    grad_clip = cfg["training"].get("grad_clip", 1.0)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, y_true, y_pred = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(y_true)
        train_f1    = compute_macro_f1(y_true, y_pred)

        val_loss, val_f1, vt, vp = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d} | "
            f"train loss={train_loss:.4f} f1={train_f1:.4f} | "
            f"val loss={val_loss:.4f} f1={val_f1:.4f}"
        )

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "best_f1": best_f1},
                str(ckpt_dir / "image_best.pt"),
            )

        # Save latest
        save_checkpoint(
            {"epoch": epoch + 1, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            str(ckpt_dir / "image_last.pt"),
        )

    print(f"\nBest Val Macro-F1: {best_f1:.4f}")

    # Final evaluation with best checkpoint
    load_checkpoint(str(ckpt_dir / "image_best.pt"), model, device=device)
    _, _, vt, vp = evaluate(model, val_loader, criterion, device)
    print_metrics(vt, vp, "val (best checkpoint)")


if __name__ == "__main__":
    main()
