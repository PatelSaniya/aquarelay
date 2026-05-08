"""
image_model.py — Per-frame CNN encoder + temporal attention over sequence.

Pipeline:
  (B, T, C, H, W)
      → shared MobileNetV3 backbone   (B*T, embed_dim)
      → reshape                       (B, T, embed_dim)
      → Temporal Transformer / BiLSTM (B, T, embed_dim)
      → attention pool                (B, embed_dim)
      → Linear classifier             (B, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ──────────────────────────────────────────────────────────────────────────────
# Shared CNN backbone
# ──────────────────────────────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """
    Wraps a torchvision model, strips its classifier, adds a linear projector.
    Handles tiny 32×32 inputs.
    """

    def __init__(self, name: str = "mobilenet_v3_small", pretrained: bool = True, out_dim: int = 256, in_channels: int = 3):
        super().__init__()
        weights_arg = "DEFAULT" if pretrained else None

        if name == "mobilenet_v3_small":
            base = tvm.mobilenet_v3_small(weights=weights_arg)
            feat_dim = base.classifier[0].in_features
            base.classifier = nn.Identity()
        elif name == "efficientnet_b0":
            base = tvm.efficientnet_b0(weights=weights_arg)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
        elif name == "convnext_tiny":
            base = tvm.convnext_tiny(weights=weights_arg)
            feat_dim = base.classifier[2].in_features
            base.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {name}")

        # If input is not 3-channel (e.g., 6 with frame diff), patch first conv
        if in_channels != 3:
            first_conv = list(base.features.children())[0]
            if isinstance(first_conv, nn.Sequential):
                old = first_conv[0]
            else:
                old = first_conv
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old.weight
                if in_channels > 3:
                    new_conv.weight[:, 3:] = old.weight[:, :in_channels - 3]
            if isinstance(first_conv, nn.Sequential):
                first_conv[0] = new_conv
            else:
                base.features[0] = new_conv

        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        feats = self.backbone(x)   # (N, feat_dim)
        return self.proj(feats)    # (N, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal modules
# ──────────────────────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPool(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pos_embed[:, :T, :]
        x = self.encoder(x, src_key_padding_mask=mask)  # (B, T, D)
        return self.attn_pool(x)                        # (B, D)


class BiLSTMTemporal(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attn_pool = AttentionPool(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(x)   # (B, T, hidden_dim)
        return self.attn_pool(out)


class AttentionPool(nn.Module):
    """Weighted sum of token embeddings."""
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * w).sum(dim=1)               # (B, D)


# ──────────────────────────────────────────────────────────────────────────────
# Full image-sequence model
# ──────────────────────────────────────────────────────────────────────────────

class ImageSequenceModel(nn.Module):
    """
    Full model: CNN backbone shared across frames + temporal encoder.
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        embed_dim: int = 256,
        temporal: str = "transformer",
        num_temporal_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        use_frame_diff: bool = True,
        max_frames: int = 32,
    ):
        super().__init__()
        self.use_frame_diff = use_frame_diff
        in_channels = 6 if use_frame_diff else 3

        self.cnn = CNNBackbone(
            name=backbone, pretrained=pretrained,
            out_dim=embed_dim, in_channels=in_channels,
        )

        if temporal == "transformer":
            self.temporal = TemporalTransformer(
                d_model=embed_dim, num_heads=4,
                num_layers=num_temporal_layers,
                dropout=dropout, max_len=max_frames,
            )
        else:
            self.temporal = BiLSTMTemporal(embed_dim, embed_dim, num_temporal_layers, dropout)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)   C=3 (RGB)
        Returns: (B, embed_dim)
        """
        B, T, C, H, W = x.shape

        # Frame differencing
        if self.use_frame_diff:
            diff = torch.zeros_like(x)
            diff[:, 1:] = x[:, 1:] - x[:, :-1]
            x = torch.cat([x, diff], dim=2)  # (B, T, 6, H, W)
            C = 6

        # Flatten batch & time for CNN
        x_flat = x.reshape(B * T, C, H, W)
        feats = self.cnn(x_flat)              # (B*T, embed_dim)
        feats = feats.reshape(B, T, -1)       # (B, T, embed_dim)

        # Temporal aggregation
        return self.temporal(feats)           # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(x)
        return self.head(emb)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ImageSequenceModel()
    x = torch.randn(2, 16, 3, 32, 32)
    logits = model(x)
    print("ImageSequenceModel output:", logits.shape)  # (2, 3)
