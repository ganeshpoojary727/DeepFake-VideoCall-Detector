"""Reusable video spatial feature extractor wrapper module."""

from __future__ import annotations

from typing import Union
import torch
import torch.nn as nn

from app.video.models.efficientnet.backbone import EfficientNetB4Backbone


class FeatureExtractor(nn.Module):
    """Spatial feature extractor wrapping EfficientNet-B4 for 4D and 5D input tensors."""

    def __init__(self, backbone: Union[EfficientNetB4Backbone, nn.Module]) -> None:
        super().__init__()
        self.backbone = backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from 4D image or 5D video tensor.

        Args:
            x: Tensor of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            torch.Tensor: Feature embeddings of shape (B, 1792) or (B, T, 1792).
        """
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x_reshaped = x.view(b * t, c, h, w)
            # Process in chunks of 64 frames to fully utilize 4-5GB VRAM on RTX 4050
            chunk_size = 64
            feats_list = []
            for i in range(0, b * t, chunk_size):
                chunk = x_reshaped[i : i + chunk_size]
                if hasattr(self.backbone, "extract_features"):
                    f = self.backbone.extract_features(chunk)
                else:
                    f = self.backbone(chunk)
                feats_list.append(f)
            feats = torch.cat(feats_list, dim=0)
            feat_dim = feats.size(-1)
            return feats.view(b, t, feat_dim)
        elif x.ndim == 4:
            if hasattr(self.backbone, "extract_features"):
                return self.backbone.extract_features(x)
            return self.backbone(x)
        else:
            raise ValueError(f"Expected 4D or 5D tensor input, got shape {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to extract_features."""
        return self.extract_features(x)
