"""Video model factory module."""

from __future__ import annotations

from typing import Optional, Type
import torch
import torch.nn as nn

from app.video.configs.model_config import ModelConfig
from app.video.exceptions.video_exceptions import ModelError
from app.video.models.attention.base_attention import (
    BaseTemporalAttention,
    DummyTemporalAttention,
    attention_registry,
)
from app.video.models.backbones.base_backbone import (
    BaseBackbone,
    DummyBackbone,
    backbone_registry,
)
from app.video.models.base_video_model import BaseVideoModel
from app.video.models.classifiers.base_classifier import (
    BaseClassifier,
    LinearClassifier,
    classifier_registry,
)
from app.video.models.model_registry import model_registry


class ModularVideoModel(BaseVideoModel):
    """Assembles a video detector model from backbone, attention, and classifier components."""

    def __init__(
        self,
        backbone: BaseBackbone,
        attention: BaseTemporalAttention,
        classifier: BaseClassifier,
        config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(config=config)
        self.backbone = backbone
        self.attention = attention
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass [B, T, C, H, W] -> [B, num_classes]."""
        if x.dim() == 4:
            x = x.unsqueeze(0)  # [1, T, C, H, W]

        b, t, c, h, w = x.shape
        frames_flat = x.view(b * t, c, h, w)

        # Extract spatial features for each frame
        spatial_feats = self.backbone(frames_flat)  # [B*T, feature_dim]
        spatial_feats = spatial_feats.view(b, t, -1)  # [B, T, feature_dim]

        # Aggregate across temporal dimension
        temp_feats = self.attention(spatial_feats)  # [B, out_dim]

        # Classify
        logits = self.classifier(temp_feats)  # [B, num_classes]
        return logits


class VideoFactory:
    """Factory for building complete Video AI model instances from config."""

    @classmethod
    def create_model(cls, config: Optional[ModelConfig] = None) -> BaseVideoModel:
        """Create video model from ModelConfig.

        Args:
            config: Model configuration dataclass.

        Returns:
            BaseVideoModel: Constructed PyTorch model.
        """
        cfg = config or ModelConfig()
        cfg.validate()

        # Retrieve or instantiate backbone
        try:
            backbone_cls = backbone_registry.get(cfg.backbone_name)
            backbone = backbone_cls(in_channels=cfg.in_channels, feature_dim=cfg.feature_dim)
        except Exception:
            backbone = DummyBackbone(in_channels=cfg.in_channels, feature_dim=cfg.feature_dim)

        # Retrieve or instantiate attention
        att_name = cfg.attention_name or "dummy_attention"
        try:
            attention_cls = attention_registry.get(att_name)
            attention = attention_cls(feature_dim=cfg.feature_dim, out_dim=512)
        except Exception:
            attention = DummyTemporalAttention(feature_dim=cfg.feature_dim, out_dim=512)

        # Retrieve or instantiate classifier
        try:
            classifier_cls = classifier_registry.get(cfg.classifier_name)
            classifier = classifier_cls(in_features=512, num_classes=cfg.num_classes, dropout=cfg.dropout)
        except Exception:
            classifier = LinearClassifier(in_features=512, num_classes=cfg.num_classes, dropout=cfg.dropout)

        model = ModularVideoModel(
            backbone=backbone,
            attention=attention,
            classifier=classifier,
            config=cfg,
        )

        return model


model_registry.register("modular_video_model", ModularVideoModel, overwrite=True)
