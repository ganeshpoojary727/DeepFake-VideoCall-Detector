"""Production EfficientNet-B4 spatial feature extraction backbone module."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from app.video.core.base_video_model import BaseVideoModel

logger = logging.getLogger(__name__)


class EfficientNetB4Backbone(BaseVideoModel):
    """Production EfficientNet-B4 feature extraction backbone (1792-dim output)."""

    def __init__(
        self,
        pretrained: bool = True,
        in_channels: int = 3,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.feature_dim = 1792

        try:
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.net = efficientnet_b4(weights=weights)
        except Exception as err:
            logger.warning(f"Could not load torchvision pretrained weights: {err}. Initializing randomly.")
            self.net = efficientnet_b4(weights=None)

        self.net.classifier = nn.Identity()

        if in_channels != 3:
            old_conv = self.net.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            self.net.features[0][0] = new_conv

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Log structure details
        logger.info(
            f"Initialized EfficientNetB4Backbone (pretrained={pretrained}, "
            f"params={self.get_num_parameters():,}, trainable={self.get_trainable_parameters():,})"
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1792-dimensional pooled feature embeddings.

        Args:
            x: Input image batch tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature embeddings of shape (B, 1792).
        """
        is_frozen = not any(p.requires_grad for p in self.net.parameters())
        if is_frozen:
            was_training = self.net.training
            self.net.eval()
            with torch.no_grad():
                feats = self.net.features(x)
                pooled = self.pool(feats)
                out = torch.flatten(pooled, 1)
            if was_training:
                self.net.train()
            return out
        else:
            feats = self.net.features(x)
            pooled = self.pool(feats)
            return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to extract_features."""
        return self.extract_features(x)

    def freeze(self) -> None:
        """Freeze all parameters in the backbone."""
        for param in self.net.parameters():
            param.requires_grad = False
        logger.info("EfficientNetB4Backbone: all layers frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all parameters in the backbone."""
        for param in self.net.parameters():
            param.requires_grad = True
        logger.info("EfficientNetB4Backbone: all layers unfrozen.")

    def freeze_layers(self, until_stage: int = 4) -> None:
        """Freeze initial stages for partial fine-tuning."""
        total_stages = len(self.net.features)
        cutoff = min(max(0, until_stage), total_stages)
        for i in range(cutoff):
            for param in self.net.features[i].parameters():
                param.requires_grad = False
        logger.info(f"EfficientNetB4Backbone: frozen stages 0..{cutoff-1}.")

    def get_num_parameters(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx_placeholder(self, dummy_input: torch.Tensor, export_path: str) -> None:
        """Hook for future ONNX graph export."""
        logger.info(f"ONNX export placeholder called for path: {export_path}")

    def export_tensorrt_placeholder(self, export_path: str) -> None:
        """Hook for future TensorRT engine export."""
        logger.info(f"TensorRT export placeholder called for path: {export_path}")
