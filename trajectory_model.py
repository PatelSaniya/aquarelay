"""
trajectory_model.py — Transformer encoder over raw trajectory sequences.

Two separate encoders:
  - Traj2DEncoder: (B, T, 3)  → (B, embed_dim)    [t, h_angle, v_angle]
  - Traj3DEncoder: (B, T, 4)  → (B, embed_dim)    [t, x, y, z]

Can be used standalone or as components of the fusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding for real-valued time index
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ──────────────────────────────────────────────────────────────────────────────
# Generic trajectory encoder
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryEncoder(nn.Module):
    """
    Encodes a variable-length trajectory into a fixed-size embedding.

    Args:
        in_dim    : feature dimension per timestep (3 for 2D, 4 for 3D)
        embed_dim : output embedding size
        num_heads : transformer attention heads
        num_layers: transformer encoder depth
        dropout   : dropout probability
        max_len   : maximum sequence length (for positional encoding)
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_len: int = 256,
    ):
        super().__init__()

        # Input projection: raw features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Velocity features (appended before projection if desired)
        # We compute them inline to add motion as extra input signal
        self.vel_proj = nn.Sequential(
            nn.Linear(in_dim - 1, embed_dim // 4),   # exclude time dim
            nn.GELU(),
        )
        self.merge = nn.Linear(embed_dim + embed_dim // 4, embed_dim)

        self.pe = SinusoidalPE(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling: learn which timesteps are most discriminative
        self.attn_pool = nn.Linear(embed_dim, 1)

        self.norm = nn.LayerNorm(embed_dim)

    def _compute_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)  where x[:,:,0] = time
        Returns (B, T, in_dim-1) velocity features (padded with zero at t=0)
        """
        positions = x[:, :, 1:]              # (B, T, in_dim-1)
        vel = torch.zeros_like(positions)
        vel[:, 1:] = positions[:, 1:] - positions[:, :-1]
        return vel

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        padding_mask: (B, T) bool — True for padded positions
        Returns: (B, embed_dim)
        """
        # Base projection
        base = self.input_proj(x)                   # (B, T, embed_dim)

        # Velocity features
        vel = self._compute_velocity(x)             # (B, T, in_dim-1)
        vel_emb = self.vel_proj(vel)                # (B, T, embed_dim//4)

        # Merge
        merged = self.merge(torch.cat([base, vel_emb], dim=-1))  # (B, T, embed_dim)

        # Positional encoding
        merged = self.pe(merged)

        # Transformer
        out = self.encoder(merged, src_key_padding_mask=padding_mask)  # (B, T, embed_dim)

        # Mask padded positions before pooling
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Attention pooling
        attn_w = torch.softmax(
            self.attn_pool(out).masked_fill(
                padding_mask.unsqueeze(-1) if padding_mask is not None else torch.zeros(1, dtype=torch.bool),
                -1e9,
            ),
            dim=1,
        )  # (B, T, 1)
        pooled = (out * attn_w).sum(dim=1)          # (B, embed_dim)

        return self.norm(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────────────────

class Traj2DEncoder(TrajectoryEncoder):
    def __init__(self, embed_dim: int = 128, **kwargs):
        super().__init__(in_dim=3, embed_dim=embed_dim, **kwargs)


class Traj3DEncoder(TrajectoryEncoder):
    def __init__(self, embed_dim: int = 128, **kwargs):
        super().__init__(in_dim=4, embed_dim=embed_dim, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone trajectory-only classifier
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryOnlyModel(nn.Module):
    """
    Classifies using 2D + 3D trajectories only (no images).
    Useful as a strong baseline before adding visual features.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        max_traj_len: int = 128,
    ):
        super().__init__()

        self.enc_2d = Traj2DEncoder(embed_dim=embed_dim, num_heads=num_heads,
                                     num_layers=num_layers, dropout=dropout,
                                     max_len=max_traj_len)
        self.enc_3d = Traj3DEncoder(embed_dim=embed_dim, num_heads=num_heads,
                                     num_layers=num_layers, dropout=dropout,
                                     max_len=max_traj_len)

        fused_dim = embed_dim * 2

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def get_padding_mask(self, traj: torch.Tensor) -> torch.Tensor:
        """Returns True for rows that are all-zero (padding)."""
        return (traj.abs().sum(dim=-1) == 0)  # (B, T)

    def get_embedding(self, traj_2d: torch.Tensor, traj_3d: torch.Tensor) -> torch.Tensor:
        mask_2d = self.get_padding_mask(traj_2d)
        mask_3d = self.get_padding_mask(traj_3d)

        emb_2d = self.enc_2d(traj_2d, mask_2d)   # (B, embed_dim)
        emb_3d = self.enc_3d(traj_3d, mask_3d)   # (B, embed_dim)

        return torch.cat([emb_2d, emb_3d], dim=-1)  # (B, 2*embed_dim)

    def forward(self, traj_2d: torch.Tensor, traj_3d: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(traj_2d, traj_3d)
        return self.head(emb)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = TrajectoryOnlyModel()
    t2d = torch.randn(2, 128, 3)
    t3d = torch.randn(2, 128, 4)
    out = model(t2d, t3d)
    print("TrajectoryOnlyModel output:", out.shape)  # (2, 3)
