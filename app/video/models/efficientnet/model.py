"""Production EfficientNet-B4 complete deepfake detection model module."""

from __future__ import annotations

import enum
import logging
from typing import Dict, Optional, Union
import torch
import torch.nn as nn

from app.video.configs.model_config import ModelConfig
from app.video.core.base_video_model import BaseVideoModel
from app.video.models.attention.temporal_encoder import TemporalEncoder
from app.video.models.classifiers.classifier_head import ModularClassifierHead
from app.video.models.efficientnet.backbone import EfficientNetB4Backbone
from app.video.models.efficientnet.feature_extractor import FeatureExtractor
from app.video.models.weight_loader import WeightLoader

logger = logging.getLogger(__name__)


class ExecutionMode(enum.Enum):
    """Model execution mode."""

    TRAINING = "training"
    EVALUATION = "evaluation"
    FEATURE_EXTRACTION = "feature_extraction"
    INFERENCE = "inference"


class EfficientNetB4Model(BaseVideoModel):
    """Production EfficientNet-B4 model combining spatial backbone, temporal attention, and classifier head."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = EfficientNetB4Backbone(
            pretrained=self.config.pretrained,
            in_channels=self.config.in_channels,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )
        self.feature_extractor = FeatureExtractor(self.backbone)

        self.temporal_encoder = TemporalEncoder(
            feature_dim=self.backbone.feature_dim,
            out_dim=self.backbone.feature_dim,
            config=self.config,
        )

        self.classifier = ModularClassifierHead(
            in_features=self.backbone.feature_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
            activation_fn=self.config.activation_fn,
            norm_layer=self.config.norm_layer,
        )

        self.mode = ExecutionMode.TRAINING

        if self.config.freeze_backbone:
            self.backbone.freeze()

        if self.config.checkpoint_path is not None:
            self.load_weights(self.config.checkpoint_path)

        logger.info(
            f"Initialized EfficientNetB4Model (num_classes={self.config.num_classes}, "
            f"total_params={self.get_num_parameters():,}, trainable={self.get_trainable_parameters():,})"
        )

    def set_mode(self, mode: Union[ExecutionMode, str]) -> None:
        """Set model execution mode."""
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower().strip())
        self.mode = mode

        if mode == ExecutionMode.TRAINING:
            self.train()
        else:
            self.eval()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature embeddings without classification head."""
        return self.feature_extractor.extract_features(x)

    def extract_clip_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporally aggregated clip embedding (B, 1792)."""
        frame_feats = self.extract_features(x)
        if frame_feats.ndim == 2:
            frame_feats = frame_feats.unsqueeze(1)
        return self.temporal_encoder(frame_feats)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal attention weights for video frames.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            torch.Tensor: Attention weights of shape (B, T, 1) summing to 1.0 over T.
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
        frame_feats = self.extract_features(x)
        return self.temporal_encoder.get_attention_weights(frame_feats)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            torch.Tensor: Logits (B, num_classes) or clip embeddings (B, 1792) in FEATURE_EXTRACTION mode.
        """
        if self.mode == ExecutionMode.FEATURE_EXTRACTION:
            return self.extract_clip_embedding(x)

        if x.ndim == 4:
            x = x.unsqueeze(1)

        frame_feats = self.extract_features(x)  # (B, T, 1792)
        clip_emb = self.temporal_encoder(frame_feats)  # (B, 1792)
        return self.classifier(clip_emb)  # (B, num_classes)

    def freeze(self) -> None:
        """Freeze backbone parameters."""
        self.backbone.freeze()

    def unfreeze(self) -> None:
        """Unfreeze backbone parameters."""
        self.backbone.unfreeze()

    def load_weights(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Any]:
        """Load pretrained checkpoint weights."""
        return WeightLoader.load_weights(self, checkpoint_path=checkpoint_path, strict=strict)

    def get_num_parameters(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Aliases
EfficientNetB4Wrapper = EfficientNetB4Model
