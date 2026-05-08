"""
predict.py — Generate predictions for the test set.

Supports:
  - Single model inference (fusion or image-only)
  - Ensemble: average probabilities from trajectory (LightGBM) + fusion model

Usage:
    # Single model
    python -m src.predict --model fusion --ckpt checkpoints/fusion_best.pt

    # Ensemble
    python -m src.predict --model ensemble \
        --ckpt checkpoints/fusion_best.pt \
        --lgbm_ckpt checkpoints/lgbm_trajectory.txt

    # Validate on val split
    python -m src.predict --model fusion --ckpt checkpoints/fusion_best.pt --split val
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VisgridDataset, VisgridTestDataset, IDX_TO_CLASS
from src.models.fusion_model import build_fusion_model
from src.features import build_feature_matrix
from src.utils import load_config, print_metrics, compute_macro_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--data_root",  default=None)
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--model",      default="fusion", choices=["fusion", "image", "lgbm", "ensemble"])
    p.add_argument("--ckpt",       default="checkpoints/fusion_best.pt")
    p.add_argument("--lgbm_ckpt",  default="checkpoints/lgbm_trajectory.txt")
    p.add_argument("--ensemble_weights", default="0.4,0.6",
                   help="Comma-separated weights for lgbm,fusion probabilities")
    p.add_argument("--out",        default=None, help="Output CSV path")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Neural model inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_neural(model, loader, device):
    model.eval()
    all_probs, all_ids = [], []

    for batch in tqdm(loader, desc="  Inference"):
        batch_dev = {
            "images":  batch["images"].to(device),
            "traj_2d": batch["traj_2d"].to(device),
            "traj_3d": batch["traj_3d"].to(device),
        }
        logits = model(batch_dev)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

        # Object IDs are only present if return_id=True
        if "object_id" in batch:
            all_ids.extend(batch["object_id"])

    return np.vstack(all_probs), all_ids


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM inference
# ──────────────────────────────────────────────────────────────────────────────

def run_lgbm(lgbm_path, split_dir, labels=None):
    import lightgbm as lgb
    model = lgb.Booster(model_file=lgbm_path)
    X, y, object_ids, _ = build_feature_matrix(split_dir, labels)
    probs = model.predict(X)
    return probs, object_ids, y


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    root = Path(args.data_root or cfg["data"]["root"])

    # Resolve split directory
    if args.split == "test":
        split_dir = root / "test"
        has_labels = False
    else:
        split_dir = root / cfg["data"][f"{args.split}_dir"]
        has_labels = True

    labels = None
    if has_labels:
        with open(split_dir / "labels.json") as f:
            labels = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Split: {args.split} | Model: {args.model}")

    out_path = args.out or f"outputs/predictions_{args.split}_{args.model}.csv"
    Path(out_path).parent.mkdir(exist_ok=True)

    # ── LightGBM only ─────────────────────────────────────────────────────────
    if args.model == "lgbm":
        probs, object_ids, y = run_lgbm(args.lgbm_ckpt, str(split_dir), labels)
        preds = probs.argmax(axis=1)

    # ── Neural model (fusion or image-only) ───────────────────────────────────
    elif args.model in ("fusion", "image"):
        if has_labels:
            ds = VisgridDataset(str(split_dir), return_id=True,
                                max_frames=cfg["data"]["max_frames"],
                                max_traj_len=cfg["data"]["max_traj_len"])
            y = [ds.labels[oid] for oid in ds.object_ids]
            from src.dataset import CLASS_MAP
            y = np.array([CLASS_MAP[lbl] for lbl in y])
        else:
            ds = VisgridTestDataset(str(split_dir),
                                    max_frames=cfg["data"]["max_frames"],
                                    max_traj_len=cfg["data"]["max_traj_len"])
            y = None

        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"])

        # Build model
        nn_model = build_fusion_model(cfg).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        nn_model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint (epoch {ckpt.get('epoch','?')}, F1={ckpt.get('best_f1',0):.4f})")

        probs, object_ids = run_neural(nn_model, loader, device)
        preds = probs.argmax(axis=1)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    elif args.model == "ensemble":
        w = [float(x) for x in args.ensemble_weights.split(",")]
        w_lgbm, w_neural = w[0], w[1]

        # LightGBM
        lgbm_probs, object_ids, y = run_lgbm(args.lgbm_ckpt, str(split_dir), labels)

        # Neural
        if has_labels:
            ds = VisgridDataset(str(split_dir), return_id=True,
                                max_frames=cfg["data"]["max_frames"],
                                max_traj_len=cfg["data"]["max_traj_len"])
        else:
            ds = VisgridTestDataset(str(split_dir),
                                    max_frames=cfg["data"]["max_frames"],
                                    max_traj_len=cfg["data"]["max_traj_len"])
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"])

        nn_model = build_fusion_model(cfg).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        nn_model.load_state_dict(ckpt["model"])

        neural_probs, _ = run_neural(nn_model, loader, device)

        # Align order (LightGBM uses split object_ids order)
        probs = w_lgbm * lgbm_probs + w_neural * neural_probs
        preds = probs.argmax(axis=1)

    # ── Save output ───────────────────────────────────────────────────────────
    rows = []
    for i, oid in enumerate(object_ids):
        row = {
            "object_id":      oid,
            "prediction":     IDX_TO_CLASS[preds[i]],
            "prob_bird":      round(float(probs[i, 0]), 4),
            "prob_drone":     round(float(probs[i, 1]), 4),
            "prob_irrelevant":round(float(probs[i, 2]), 4),
        }
        if y is not None:
            row["true"] = IDX_TO_CLASS[int(y[i])]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nPredictions saved → {out_path}")
    print(df.head(10).to_string())

    # Evaluate if ground truth is available
    if y is not None:
        print_metrics(y.astype(int), preds.astype(int), args.split)

    return df


if __name__ == "__main__":
    main()
