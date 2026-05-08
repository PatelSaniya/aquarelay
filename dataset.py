"""
dataset.py — Visgrid dataset loader.

Each object has:
  - images/  : 32x32 RGB PNGs from 1–6 cameras
  - trajectories_2d/ : per-camera angular trajectories (h_angle, v_angle)
  - trajectories_3d/ : fused 3D position (x, y, z)

Label: BIRD (0), DRONE (1), IRRELEVANT (2)
"""

import os
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

CLASS_MAP = {"BIRD": 0, "DRONE": 1, "IRRELEVANT": 2}
IDX_TO_CLASS = {0: "BIRD", 1: "DRONE", 2: "IRRELEVANT"}


def _parse_timestamp(filename: str) -> int:
    """Extract millisecond timestamp from image filename like cam01_t1778232740880ms.png"""
    m = re.search(r"_t(\d+)ms", filename)
    return int(m.group(1)) if m else 0


def _parse_camera_id(filename: str) -> str:
    """Extract camera_id from filename like cam01_t...ms.png"""
    return filename.split("_t")[0]


class VisgridDataset(Dataset):
    """
    Returns per-object tensors:
      images   : (max_frames, 3, 32, 32)  float32, normalised [0,1]
      traj_2d  : (max_traj_len, 3)        float32  [t, h_angle, v_angle]
      traj_3d  : (max_traj_len, 4)        float32  [t, x, y, z]
      label    : ()                        long
    """

    def __init__(
        self,
        split_dir: str,
        max_frames: int = 32,
        max_traj_len: int = 128,
        transform=None,
        return_id: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.max_frames = max_frames
        self.max_traj_len = max_traj_len
        self.transform = transform
        self.return_id = return_id

        with open(self.split_dir / "labels.json") as f:
            self.labels = json.load(f)

        self.object_ids = sorted(self.labels.keys())
        self.images_dir = self.split_dir / "images"
        self.traj_2d_dir = self.split_dir / "trajectories_2d"
        self.traj_3d_dir = self.split_dir / "trajectories_3d"

    def __len__(self):
        return len(self.object_ids)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_images(self, object_id: str) -> torch.Tensor:
        """
        Load frames sorted by timestamp.
        Groups by camera so we get a consistent temporal sequence.
        Returns (max_frames, 3, 32, 32).
        """
        obj_dir = self.images_dir / object_id
        if not obj_dir.exists():
            return torch.zeros(self.max_frames, 3, 32, 32)

        files = sorted(
            obj_dir.glob("*.png"),
            key=lambda p: _parse_timestamp(p.name),
        )

        frames = []
        for fp in files[: self.max_frames]:
            img = np.array(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
            # frame differencing: append diff from previous frame as extra channel
            frames.append(img)

        if not frames:
            return torch.zeros(self.max_frames, 3, 32, 32)

        # Pad with zeros to max_frames
        blank = np.zeros_like(frames[0])
        while len(frames) < self.max_frames:
            frames.append(blank)

        arr = np.stack(frames[: self.max_frames])  # (T, H, W, C)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor

    # ------------------------------------------------------------------
    # 2D trajectory loading
    # ------------------------------------------------------------------
    def _load_traj_2d(self, object_id: str) -> torch.Tensor:
        """
        Aggregate all cameras' 2D angular trajectories, sort by time.
        Returns (max_traj_len, 3):  [t_norm, h_angle, v_angle]
        """
        files = sorted(self.traj_2d_dir.glob(f"{object_id}_*.json"))

        all_pts: list[list[float]] = []
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
            base_ms = data.get("timestamp_ms", 0)
            for pt in data["points"]:
                all_pts.append([pt["t"], pt["h_angle"], pt["v_angle"]])

        return self._pack_trajectory(all_pts, n_cols=3)

    # ------------------------------------------------------------------
    # 3D trajectory loading
    # ------------------------------------------------------------------
    def _load_traj_3d(self, object_id: str) -> torch.Tensor:
        """
        Load fused 3D trajectory.
        Returns (max_traj_len, 4):  [t_norm, x, y, z]
        """
        candidates = list(self.traj_3d_dir.glob(f"{object_id}*.json"))
        if not candidates:
            return torch.zeros(self.max_traj_len, 4)

        with open(candidates[0]) as f:
            data = json.load(f)

        pts = [[p["t"], p["x"], p["y"], p["z"]] for p in data["points"]]
        return self._pack_trajectory(pts, n_cols=4)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _pack_trajectory(self, pts: list, n_cols: int) -> torch.Tensor:
        if not pts:
            return torch.zeros(self.max_traj_len, n_cols)

        arr = np.array(sorted(pts, key=lambda x: x[0]), dtype=np.float32)

        # Normalise time to [0, 1]
        t_max = arr[-1, 0]
        if t_max > 0:
            arr[:, 0] /= t_max

        # Pad / truncate
        if len(arr) >= self.max_traj_len:
            arr = arr[: self.max_traj_len]
        else:
            pad = np.zeros((self.max_traj_len - len(arr), n_cols), dtype=np.float32)
            arr = np.vstack([arr, pad])

        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        oid = self.object_ids[idx]
        label = CLASS_MAP[self.labels[oid]]

        item = {
            "images": self._load_images(oid),
            "traj_2d": self._load_traj_2d(oid),
            "traj_3d": self._load_traj_3d(oid),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.return_id:
            item["object_id"] = oid

        return item


# ------------------------------------------------------------------
# Unlabelled test dataset (no labels.json)
# ------------------------------------------------------------------
class VisgridTestDataset(VisgridDataset):
    def __init__(self, test_dir: str, **kwargs):
        self.split_dir = Path(test_dir)
        self.max_frames = kwargs.get("max_frames", 32)
        self.max_traj_len = kwargs.get("max_traj_len", 128)
        self.transform = kwargs.get("transform", None)
        self.return_id = True

        self.images_dir = self.split_dir / "images"
        self.traj_2d_dir = self.split_dir / "trajectories_2d"
        self.traj_3d_dir = self.split_dir / "trajectories_3d"

        # Discover object IDs from images folder
        self.object_ids = sorted(
            p.name for p in self.images_dir.iterdir() if p.is_dir()
        )
        self.labels = {}  # no labels

    def __getitem__(self, idx: int) -> dict:
        oid = self.object_ids[idx]
        item = {
            "images": self._load_images(oid),
            "traj_2d": self._load_traj_2d(oid),
            "traj_3d": self._load_traj_3d(oid),
            "label": torch.tensor(-1, dtype=torch.long),  # unknown
            "object_id": oid,
        }
        return item
