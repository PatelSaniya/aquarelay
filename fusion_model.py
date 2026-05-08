"""
fusion_model.py — Multi-modal fusion of image sequences + 2D/3D trajectories.

Architecture:
  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │ ImageSequenceModel│  │ Traj2DEncoder   │  │ Traj3DEncoder   │
  │ (B,T,C,H,W)→(B,D)│  │ (B,T,3)→(B,D)  │  │ (B,T,4)→(B,D)  │
  └────────┬─────────┘  └───────┬─────────┘  └───────┬─────────┘
           │                    │                     │
           └────────────────────┴─────────────────────┘
                                │
                       Cross-modal attention
                                │
                         MLP Classifier → 3 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.image_model import ImageSequenceModel, AttentionPool
from src.models.trajectory_model import Traj2DEncoder, Traj3DEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Cross-modal attention (optional but recommended)
# ──────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Soft attention over modality embeddings.
    Learns which modality is most informative per sample.
    """

    def __init__(self, d_model: int, num_modalities: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model * num_modalities, num_modalities),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        # embeddings: list of (B, d_model) tensors
        stacked = torch.stack(embeddings, dim=1)      # (B, M, D)
        B, M, D = stacked.shape
        cat = stacked.reshape(B, M * D)
        scores = torch.softmax(self.scorer(cat), dim=-1)  # (B, M)
        out = (stacked * scores.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Fusion model
# ──────────────────────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Full multi-modal classifier.

    Args:
        image_cfg  : dict of kwargs for ImageSequenceModel
        traj_cfg   : dict of kwargs for trajectory encoders
        hidden_dim : MLP hidden size
        dropout    : dropout probability
        num_classes: number of output classes
        use_image  : include image modality
        use_traj_2d: include 2D trajectory modality
        use_traj_3d: include 3D trajectory modality
    """

    def __init__(
        self,
        image_cfg: dict = None,
        traj_cfg: dict = None,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 3,
        use_image: bool = True,
        use_traj_2d: bool = True,
        use_traj_3d: bool = True,
    ):
        super().__init__()

        self.use_image   = use_image
        self.use_traj_2d = use_traj_2d
        self.use_traj_3d = use_traj_3d

        image_cfg  = image_cfg  or {}
        traj_cfg   = traj_cfg   or {}

        img_embed_dim  = image_cfg.get("embed_dim", 256)
        traj_embed_dim = traj_cfg.get("embed_dim", 128)

        # ── Sub-encoders ──────────────────────────────────────────────────────
        if use_image:
            self.image_encoder = ImageSequenceModel(
                backbone=image_cfg.get("backbone", "mobilenet_v3_small"),
                pretrained=image_cfg.get("pretrained", True),
                embed_dim=img_embed_dim,
                temporal=image_cfg.get("temporal", "transformer"),
                num_temporal_layers=image_cfg.get("num_temporal_layers", 2),
                dropout=image_cfg.get("dropout", dropout),
                use_frame_diff=image_cfg.get("use_frame_diff", True),
                max_frames=image_cfg.get("max_frames", 32),
                num_classes=num_classes,      # unused (we take embedding)
            )

        if use_traj_2d:
            self.traj_2d_encoder = Traj2DEncoder(
                embed_dim=traj_embed_dim,
                num_heads=traj_cfg.get("num_heads", 4),
                num_layers=traj_cfg.get("num_layers", 3),
                dropout=traj_cfg.get("dropout", 0.2),
                max_len=traj_cfg.get("max_traj_len", 128),
            )

        if use_traj_3d:
            self.traj_3d_encoder = Traj3DEncoder(
                embed_dim=traj_embed_dim,
                num_heads=traj_cfg.get("num_heads", 4),
                num_layers=traj_cfg.get("num_layers", 3),
                dropout=traj_cfg.get("dropout", 0.2),
                max_len=traj_cfg.get("max_traj_len", 128),
            )

        # ── Fusion dims ───────────────────────────────────────────────────────
        total_dim = 0
        if use_image:   total_dim += img_embed_dim
        if use_traj_2d: total_dim += traj_embed_dim
        if use_traj_3d: total_dim += traj_embed_dim

        # Optional cross-modal attention
        num_modalities = sum([use_image, use_traj_2d, use_traj_3d])
        single_dim = img_embed_dim if use_image else traj_embed_dim

        # Project all to the same dim before cross-attention
        if use_image and (use_traj_2d or use_traj_3d):
            self.img_proj  = nn.Linear(img_embed_dim,  single_dim)
            self.t2d_proj  = nn.Linear(traj_embed_dim, single_dim) if use_traj_2d else None
            self.t3d_proj  = nn.Linear(traj_embed_dim, single_dim) if use_traj_3d else None
            self.cross_attn = CrossModalAttention(single_dim, num_modalities)
            fusion_in = total_dim + single_dim   # concat + attended
        else:
            self.cross_attn = None
            fusion_in = total_dim

        # ── MLP head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _padding_mask(traj: torch.Tensor) -> torch.Tensor:
        return (traj.abs().sum(dim=-1) == 0)  # (B, T)

    # ── Forward ───────────────────────────────────────────────────────────────

    def get_embeddings(self, batch: dict) -> torch.Tensor:
        parts = []
        proj_parts = []

        if self.use_image:
            img_emb = self.image_encoder.get_embedding(batch["images"])
            parts.append(img_emb)
            proj_parts.append(self.img_proj(img_emb))

        if self.use_traj_2d:
            t2d = batch["traj_2d"]
            mask_2d = self._padding_mask(t2d)
            t2d_emb = self.traj_2d_encoder(t2d, mask_2d)
            parts.append(t2d_emb)
            if self.t2d_proj:
                proj_parts.append(self.t2d_proj(t2d_emb))

        if self.use_traj_3d:
            t3d = batch["traj_3d"]
            mask_3d = self._padding_mask(t3d)
            t3d_emb = self.traj_3d_encoder(t3d, mask_3d)
            parts.append(t3d_emb)
            if self.t3d_proj:
                proj_parts.append(self.t3d_proj(t3d_emb))

        cat_emb = torch.cat(parts, dim=-1)   # (B, total_dim)

        if self.cross_attn is not None and len(proj_parts) > 1:
            attended = self.cross_attn(proj_parts)  # (B, single_dim)
            return torch.cat([cat_emb, attended], dim=-1)

        return cat_emb

    def forward(self, batch: dict) -> torch.Tensor:
        emb = self.get_embeddings(batch)
        return self.head(emb)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_fusion_model(cfg: dict) -> FusionModel:
    return FusionModel(
        image_cfg={
            "backbone":           cfg["image_model"]["backbone"],
            "pretrained":         cfg["image_model"]["pretrained"],
            "embed_dim":          cfg["image_model"]["embed_dim"],
            "temporal":           cfg["image_model"]["temporal"],
            "num_temporal_layers":cfg["image_model"]["num_temporal_layers"],
            "dropout":            cfg["image_model"]["dropout"],
            "max_frames":         cfg["data"]["max_frames"],
        },
        traj_cfg={
            "embed_dim":          cfg["trajectory_model"]["embed_dim"],
            "num_heads":          cfg["trajectory_model"]["num_heads"],
            "num_layers":         cfg["trajectory_model"]["num_layers"],
            "dropout":            cfg["trajectory_model"]["dropout"],
            "max_traj_len":       cfg["data"]["max_traj_len"],
        },
        hidden_dim=cfg["fusion"]["hidden_dim"],
        dropout=cfg["fusion"]["dropout"],
        use_image=cfg["fusion"]["use_image"],
        use_traj_2d=cfg["fusion"]["use_traj_2d"],
        use_traj_3d=cfg["fusion"]["use_traj_3d"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FusionModel()
    batch = {
        "images":   torch.randn(2, 32, 3, 32, 32),
        "traj_2d":  torch.randn(2, 128, 3),
        "traj_3d":  torch.randn(2, 128, 4),
    }
    out = model(batch)
    print("FusionModel output:", out.shape)   # (2, 3)
