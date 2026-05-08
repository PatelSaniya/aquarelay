"""
eda.py — Exploratory data analysis of the Visgrid dataset.

Generates:
  - Class distribution plots
  - Trajectory visualisations (2D and 3D)
  - Example image sequences per class
  - Motion feature distributions per class

Usage:
    python -m src.eda
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from src.utils import load_config
from src.features import extract_object_features


def load_split(split_dir: Path):
    with open(split_dir / "labels.json") as f:
        labels = json.load(f)
    return labels


def plot_class_distribution(labels_train, labels_val, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (labels, title) in zip(axes, [(labels_train, "Train"), (labels_val, "Val")]):
        cnt = Counter(labels.values())
        classes = ["BIRD", "DRONE", "IRRELEVANT"]
        counts  = [cnt.get(c, 0) for c in classes]
        bars = ax.bar(classes, counts, color=["#4c9be8", "#e84c4c", "#4ce87a"])
        ax.set_title(f"{title} — {sum(counts)} objects")
        ax.set_ylabel("Count")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_example_sequences(split_dir: Path, labels: dict, save_path, n_per_class=5):
    classes = {"BIRD": [], "DRONE": [], "IRRELEVANT": []}
    for oid, lbl in labels.items():
        classes[lbl].append(oid)

    fig, axes = plt.subplots(
        3, n_per_class * 5,
        figsize=(n_per_class * 5 * 1.2, 7),
        gridspec_kw={"hspace": 0.4, "wspace": 0.05}
    )

    for row_idx, (cls_name, oids) in enumerate(classes.items()):
        sample_oids = random.sample(oids, min(n_per_class, len(oids)))
        for col_obj, oid in enumerate(sample_oids):
            img_dir = split_dir / "images" / oid
            if not img_dir.exists():
                continue
            frames = sorted(img_dir.glob("*.png"),
                            key=lambda p: int(p.stem.split("_t")[1].replace("ms", "")))[:5]
            for col_frame, fp in enumerate(frames):
                ax = axes[row_idx, col_obj * 5 + col_frame]
                img = np.array(Image.open(fp).convert("RGB"))
                ax.imshow(img)
                ax.axis("off")
                if col_frame == 0:
                    ax.set_title(f"{cls_name}\n{oid[:8]}", fontsize=6)

    fig.suptitle("Example sequences (5 frames per object)", fontsize=13)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_trajectory_3d(split_dir: Path, labels: dict, save_path, n_per_class=5):
    classes = {"BIRD": [], "DRONE": [], "IRRELEVANT": []}
    for oid, lbl in labels.items():
        classes[lbl].append(oid)

    colors = {"BIRD": "#4c9be8", "DRONE": "#e84c4c", "IRRELEVANT": "#4ce87a"}

    fig = plt.figure(figsize=(15, 4))
    for idx, (cls_name, oids) in enumerate(classes.items()):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_title(cls_name)
        sample_oids = random.sample(oids, min(n_per_class, len(oids)))
        for oid in sample_oids:
            traj_dir = split_dir / "trajectories_3d"
            candidates = list(traj_dir.glob(f"{oid}*.json"))
            if not candidates:
                continue
            with open(candidates[0]) as f:
                data = json.load(f)
            pts = np.array([[p["x"], p["y"], p["z"]] for p in data["points"]])
            if len(pts) < 2:
                continue
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.7, color=colors[cls_name], linewidth=1)
            ax.scatter(*pts[0], color="black", s=10, zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

    fig.suptitle("3D Trajectories per class", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_feature_distributions(split_dir: Path, labels: dict, save_path):
    """Compare key motion features across classes."""
    feat_data = defaultdict(list)
    class_list = []

    for oid, lbl in labels.items():
        feats = extract_object_features(oid, split_dir)
        feat_data[lbl].append(feats)
        class_list.append(lbl)

    key_features = [
        "3d_speed_mean", "3d_acc_std", "3d_z_std",
        "3d_hover_frac", "3d_curvature_mean", "3d_straightness",
        "2d_h_vel_std", "2d_v_fft_amp",
    ]
    colors = {"BIRD": "#4c9be8", "DRONE": "#e84c4c", "IRRELEVANT": "#4ce87a"}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for ax, feat in zip(axes, key_features):
        for cls_name, cls_feats in feat_data.items():
            vals = [f.get(feat, 0.0) for f in cls_feats]
            vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
            if not vals:
                continue
            ax.hist(vals, bins=20, alpha=0.6, color=colors[cls_name], label=cls_name, density=True)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("Motion Feature Distributions by Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def main():
    cfg = load_config("configs/config.yaml")
    root = Path(cfg["data"]["root"])
    train_dir = root / cfg["data"]["train_dir"]
    val_dir   = root / cfg["data"]["val_dir"]
    out_dir   = Path(cfg["paths"]["outputs"]) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading labels …")
    train_labels = load_split(train_dir)
    val_labels   = load_split(val_dir)

    print("\n[1] Class distribution …")
    plot_class_distribution(train_labels, val_labels, out_dir / "class_distribution.png")

    print("[2] Example sequences …")
    plot_example_sequences(train_dir, train_labels, out_dir / "example_sequences.png")

    print("[3] 3D trajectories …")
    plot_trajectory_3d(train_dir, train_labels, out_dir / "trajectories_3d.png")

    print("[4] Feature distributions …")
    plot_feature_distributions(train_dir, train_labels, out_dir / "feature_distributions.png")

    print(f"\nAll EDA plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
