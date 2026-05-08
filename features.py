"""
features.py — Hand-crafted motion features for the LightGBM baseline.

Key insight: birds and drones differ MORE in motion dynamics than appearance.

Features extracted:
  - Speed (mean, std, max, min)
  - Acceleration (mean, std, max)
  - Jerk (mean, std)
  - Path curvature
  - Altitude (z) variation and oscillation
  - Hover fraction (near-zero speed)
  - FFT peak (wingbeat periodicity)
  - 2D angular velocity (per axis)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.fft import rfft, rfftfreq
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_stat(arr: np.ndarray, func, default=0.0):
    return float(func(arr)) if len(arr) > 0 else default


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    dt = np.diff(times)
    dt = np.where(dt < 1e-9, 1e-9, dt)
    return np.diff(values, axis=0) / dt[:, None] if values.ndim == 2 else np.diff(values) / dt


def _fft_features(signal: np.ndarray, prefix: str) -> dict:
    """Return dominant frequency and its amplitude."""
    if len(signal) < 8:
        return {f"{prefix}_fft_amp": 0.0, f"{prefix}_fft_freq": 0.0}
    s = signal - signal.mean()
    fft = np.abs(rfft(s))
    fft[0] = 0  # ignore DC
    peak_idx = np.argmax(fft)
    return {
        f"{prefix}_fft_amp": float(fft[peak_idx]),
        f"{prefix}_fft_freq": float(peak_idx),
        f"{prefix}_fft_entropy": float(stats.entropy(fft + 1e-9)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3D trajectory features
# ──────────────────────────────────────────────────────────────────────────────

def extract_3d_features(pts: np.ndarray) -> dict:
    """
    pts: (N, 4) array  [t, x, y, z]
    """
    feats: dict = {}
    prefix = "3d"

    if pts is None or len(pts) < 3:
        return {f"{prefix}_missing": 1.0}

    feats[f"{prefix}_missing"] = 0.0
    feats[f"{prefix}_n_points"] = float(len(pts))

    t = pts[:, 0]
    xyz = pts[:, 1:]
    z = pts[:, 3]
    xy = pts[:, 1:3]

    # ── velocity ──────────────────────────────────────────────────────────────
    vel = _derivative(xyz, t)            # (N-1, 3)
    speed = np.linalg.norm(vel, axis=1)  # (N-1,)

    feats[f"{prefix}_speed_mean"] = _safe_stat(speed, np.mean)
    feats[f"{prefix}_speed_std"]  = _safe_stat(speed, np.std)
    feats[f"{prefix}_speed_max"]  = _safe_stat(speed, np.max)
    feats[f"{prefix}_speed_min"]  = _safe_stat(speed, np.min)
    feats[f"{prefix}_speed_cv"]   = (_safe_stat(speed, np.std) /
                                      (_safe_stat(speed, np.mean) + 1e-9))

    # ── horizontal speed ──────────────────────────────────────────────────────
    vel_h = _derivative(xy, t)
    speed_h = np.linalg.norm(vel_h, axis=1)
    feats[f"{prefix}_speed_h_mean"] = _safe_stat(speed_h, np.mean)
    feats[f"{prefix}_speed_h_std"]  = _safe_stat(speed_h, np.std)

    # ── vertical velocity ─────────────────────────────────────────────────────
    vz = _derivative(z, t)
    feats[f"{prefix}_vz_mean"]     = _safe_stat(vz, np.mean)
    feats[f"{prefix}_vz_std"]      = _safe_stat(vz, np.std)
    feats[f"{prefix}_vz_abs_mean"] = _safe_stat(np.abs(vz), np.mean)

    # ── acceleration ──────────────────────────────────────────────────────────
    if len(vel) > 2:
        acc = _derivative(vel, t[:-1])
        acc_mag = np.linalg.norm(acc, axis=1)
        feats[f"{prefix}_acc_mean"] = _safe_stat(acc_mag, np.mean)
        feats[f"{prefix}_acc_std"]  = _safe_stat(acc_mag, np.std)
        feats[f"{prefix}_acc_max"]  = _safe_stat(acc_mag, np.max)

        # jerk
        if len(acc) > 2:
            jerk = _derivative(acc, t[:-2])
            jerk_mag = np.linalg.norm(jerk, axis=1)
            feats[f"{prefix}_jerk_mean"] = _safe_stat(jerk_mag, np.mean)
            feats[f"{prefix}_jerk_std"]  = _safe_stat(jerk_mag, np.std)

    # ── altitude ──────────────────────────────────────────────────────────────
    feats[f"{prefix}_z_mean"]  = float(z.mean())
    feats[f"{prefix}_z_std"]   = float(z.std())
    feats[f"{prefix}_z_range"] = float(z.max() - z.min())

    # ── hover fraction ────────────────────────────────────────────────────────
    hover_thresh = max(speed.mean() * 0.2, 0.05)
    feats[f"{prefix}_hover_frac"] = float((speed < hover_thresh).mean())

    # ── curvature (cross product of consecutive velocity vectors) ─────────────
    if len(vel) > 2:
        cross = np.cross(vel[:-1], vel[1:])          # may be scalar or 3-vector
        curvature = np.linalg.norm(cross, axis=-1) if cross.ndim > 1 else np.abs(cross)
        speed_prod = speed[:-1] * speed[1:] + 1e-9
        norm_curv = curvature / speed_prod
        feats[f"{prefix}_curvature_mean"] = _safe_stat(norm_curv, np.mean)
        feats[f"{prefix}_curvature_std"]  = _safe_stat(norm_curv, np.std)

    # ── path length ───────────────────────────────────────────────────────────
    step_len = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    feats[f"{prefix}_path_len"] = float(step_len.sum())
    displacement = np.linalg.norm(xyz[-1] - xyz[0])
    feats[f"{prefix}_straightness"] = displacement / (feats[f"{prefix}_path_len"] + 1e-9)

    # ── FFT (altitude oscillation – wing-beat proxy) ──────────────────────────
    feats.update(_fft_features(z, f"{prefix}_z"))

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# 2D trajectory features (per camera, then aggregated)
# ──────────────────────────────────────────────────────────────────────────────

def extract_2d_features(trajs: list[np.ndarray]) -> dict:
    """
    trajs: list of (N, 3) arrays  [t, h_angle, v_angle]
    """
    feats: dict = {}
    prefix = "2d"

    if not trajs:
        return {f"{prefix}_missing": 1.0, f"{prefix}_n_cameras": 0}

    feats[f"{prefix}_missing"]   = 0.0
    feats[f"{prefix}_n_cameras"] = float(len(trajs))

    # Aggregate all cameras
    all_h = np.concatenate([tr[:, 1] for tr in trajs])
    all_v = np.concatenate([tr[:, 2] for tr in trajs])
    all_t = np.concatenate([tr[:, 0] for tr in trajs])
    sort_idx = np.argsort(all_t)
    all_h = all_h[sort_idx]
    all_v = all_v[sort_idx]
    all_t = all_t[sort_idx]

    for name, ang in [("h", all_h), ("v", all_v)]:
        feats[f"{prefix}_{name}_mean"]  = float(ang.mean())
        feats[f"{prefix}_{name}_std"]   = float(ang.std())
        feats[f"{prefix}_{name}_range"] = float(ang.max() - ang.min())

        if len(all_t) > 1:
            dt = np.diff(all_t)
            dt = np.where(dt < 1e-9, 1e-9, dt)
            vel = np.diff(ang) / dt
            feats[f"{prefix}_{name}_vel_mean"]     = float(vel.mean())
            feats[f"{prefix}_{name}_vel_std"]      = float(vel.std())
            feats[f"{prefix}_{name}_vel_abs_mean"] = float(np.abs(vel).mean())
            feats[f"{prefix}_{name}_vel_max"]      = float(np.abs(vel).max())

            if len(vel) > 1:
                acc = np.diff(vel) / dt[:-1]
                feats[f"{prefix}_{name}_acc_std"] = float(acc.std())

        feats.update(_fft_features(ang, f"{prefix}_{name}"))

    # Cross-axis correlation
    if len(all_h) > 1:
        feats[f"{prefix}_hv_corr"] = float(np.corrcoef(all_h, all_v)[0, 1]) if len(all_h) > 2 else 0.0

    # Per-camera consistency
    if len(trajs) > 1:
        h_stds = [float(tr[:, 1].std()) for tr in trajs]
        v_stds = [float(tr[:, 2].std()) for tr in trajs]
        feats[f"{prefix}_inter_cam_h_std"] = float(np.std(h_stds))
        feats[f"{prefix}_inter_cam_v_std"] = float(np.std(v_stds))

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Top-level: extract features for one object from disk
# ──────────────────────────────────────────────────────────────────────────────

def extract_object_features(object_id: str, split_dir: Path) -> dict:
    """
    Extract all features for a single object.  Returns a flat dict.
    """
    traj_2d_dir = split_dir / "trajectories_2d"
    traj_3d_dir = split_dir / "trajectories_3d"

    # ── 2D trajectories ───────────────────────────────────────────────────────
    trajs_2d: list[np.ndarray] = []
    for fp in sorted(traj_2d_dir.glob(f"{object_id}_*.json")):
        with open(fp) as f:
            data = json.load(f)
        pts = [[p["t"], p["h_angle"], p["v_angle"]] for p in data["points"]]
        if pts:
            trajs_2d.append(np.array(pts, dtype=np.float32))

    feats_2d = extract_2d_features(trajs_2d)

    # ── 3D trajectory ─────────────────────────────────────────────────────────
    candidates = list(traj_3d_dir.glob(f"{object_id}*.json"))
    pts_3d = None
    if candidates:
        with open(candidates[0]) as f:
            data = json.load(f)
        pts_3d = np.array(
            [[p["t"], p["x"], p["y"], p["z"]] for p in data["points"]],
            dtype=np.float32,
        )

    feats_3d = extract_3d_features(pts_3d)

    return {"object_id": object_id, **feats_2d, **feats_3d}


def build_feature_matrix(split_dir: str, labels: Optional[dict] = None):
    """
    Build feature matrix for all objects in a split.
    Returns (X: np.ndarray, y: np.ndarray or None, object_ids: list, feature_names: list)
    """
    from src.dataset import CLASS_MAP
    import pandas as pd

    split_path = Path(split_dir)
    if labels is not None:
        object_ids = sorted(labels.keys())
    else:
        # Infer from directory
        img_dir = split_path / "images"
        object_ids = sorted(p.name for p in img_dir.iterdir() if p.is_dir())

    rows = []
    for oid in object_ids:
        row = extract_object_features(oid, split_path)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("object_id")
    df = df.fillna(0.0)

    feature_names = list(df.columns)
    X = df.values.astype(np.float32)
    y = None
    if labels is not None:
        y = np.array([CLASS_MAP[labels[oid]] for oid in object_ids], dtype=np.int64)

    return X, y, object_ids, feature_names
