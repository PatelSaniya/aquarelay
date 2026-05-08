"""
train_trajectory.py — Phase 1: LightGBM on hand-crafted trajectory features.

This is your fastest, often surprisingly strong, baseline.
Run this first to understand the signal in the trajectories alone.

Usage:
    python -m src.train_trajectory
"""

import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.features import build_feature_matrix
from src.utils import load_config, print_metrics, set_seed
from src.dataset import CLASS_MAP, IDX_TO_CLASS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--data_root", default=None, help="Override data root")
    p.add_argument("--cv",        action="store_true", help="Run 5-fold CV instead of val split")
    p.add_argument("--save_model",action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    root = Path(args.data_root or cfg["data"]["root"])
    train_dir = root / cfg["data"]["train_dir"]
    val_dir   = root / cfg["data"]["val_dir"]

    print("=" * 60)
    print("  Phase 1 — LightGBM Trajectory Baseline")
    print("=" * 60)

    # ── Load labels ───────────────────────────────────────────────────────────
    with open(train_dir / "labels.json") as f:
        train_labels = json.load(f)
    with open(val_dir / "labels.json") as f:
        val_labels = json.load(f)

    # ── Build feature matrices ────────────────────────────────────────────────
    print("\n[1/4] Extracting train features …")
    X_train, y_train, train_ids, feat_names = build_feature_matrix(str(train_dir), train_labels)

    print("[2/4] Extracting val features …")
    X_val, y_val, val_ids, _ = build_feature_matrix(str(val_dir), val_labels)

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"  Features: {len(feat_names)}")
    print(f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val   label dist: {dict(zip(*np.unique(y_val,   return_counts=True)))}")

    # ── Build LightGBM dataset ────────────────────────────────────────────────
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   feature_name=feat_names, reference=lgb_train)

    lgbm_cfg = cfg["lgbm"]
    params = {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "learning_rate":    lgbm_cfg["learning_rate"],
        "num_leaves":       lgbm_cfg["num_leaves"],
        "max_depth":        lgbm_cfg["max_depth"],
        "min_child_samples":lgbm_cfg["min_child_samples"],
        "subsample":        lgbm_cfg["subsample"],
        "colsample_bytree": lgbm_cfg["colsample_bytree"],
        "class_weight":     lgbm_cfg["class_weight"],
        "verbose":          -1,
        "seed":             cfg["training"]["seed"],
    }

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Training LightGBM …")
    callbacks = [
        lgb.early_stopping(lgbm_cfg["early_stopping_rounds"], verbose=True),
        lgb.log_evaluation(50),
    ]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=lgbm_cfg["n_estimators"],
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating …")
    val_probs  = model.predict(X_val)
    val_preds  = val_probs.argmax(axis=1)
    train_preds = model.predict(X_train).argmax(axis=1)

    print_metrics(y_train, train_preds, "train")
    macro_f1 = print_metrics(y_val, val_preds, "val")

    # ── Feature importance ────────────────────────────────────────────────────
    imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feat_names
    ).sort_values(ascending=False)

    print("\nTop 20 most important features:")
    print(imp.head(20).to_string())

    # Save importance plot
    out_dir = Path(cfg["paths"]["outputs"])
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    imp.head(30).plot(kind="barh", ax=ax)
    ax.set_title("LightGBM Feature Importance (gain)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "lgbm_feature_importance.png", dpi=150)
    print(f"  Saved importance plot → {out_dir}/lgbm_feature_importance.png")

    # ── Save model ────────────────────────────────────────────────────────────
    if args.save_model:
        ckpt_dir = Path(cfg["paths"]["checkpoints"])
        ckpt_dir.mkdir(exist_ok=True)
        model_path = ckpt_dir / "lgbm_trajectory.txt"
        model.save_model(str(model_path))
        print(f"  Model saved → {model_path}")

    # ── Save predictions ──────────────────────────────────────────────────────
    preds_df = pd.DataFrame({
        "object_id": val_ids,
        "pred":      [IDX_TO_CLASS[p] for p in val_preds],
        "true":      [IDX_TO_CLASS[y] for y in y_val],
        "prob_bird":      val_probs[:, 0],
        "prob_drone":     val_probs[:, 1],
        "prob_irrelevant":val_probs[:, 2],
    })
    preds_df.to_csv(out_dir / "lgbm_val_predictions.csv", index=False)
    print(f"  Predictions saved → {out_dir}/lgbm_val_predictions.csv")

    # ── Optional: Cross-validation ────────────────────────────────────────────
    if args.cv:
        print("\n  Running 5-fold cross-validation on train+val …")
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1s = []
        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_all, y_all)):
            fold_train = lgb.Dataset(X_all[tr_idx], y_all[tr_idx])
            fold_val   = lgb.Dataset(X_all[vl_idx], y_all[vl_idx], reference=fold_train)
            m = lgb.train(params, fold_train, num_boost_round=lgbm_cfg["n_estimators"],
                          valid_sets=[fold_val],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            preds = m.predict(X_all[vl_idx]).argmax(axis=1)
            f1 = f1_score(y_all[vl_idx], preds, average="macro", zero_division=0)
            cv_f1s.append(f1)
            print(f"    Fold {fold+1}:  Macro-F1 = {f1:.4f}")
        print(f"\n  CV Mean: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

    return macro_f1


if __name__ == "__main__":
    main()
