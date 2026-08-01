"""Fluent builder pattern module for video subsystem components."""

from __future__ import annotations

from typing import Optional
import torch.nn as nn

from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.models.base_video_model import BaseVideoModel
from app.video.models.video_factory import VideoFactory
from app.video.pipeline.inference_pipeline import InferencePipeline
from app.video.pipeline.training_pipeline import TrainingPipeline


class VideoPipelineBuilder:
    """Fluent builder for constructing Video AI training and inference pipelines."""

    def __init__(self) -> None:
        self._model_config: Optional[ModelConfig] = None
        self._training_config: Optional[VideoTrainingConfig] = None
        self._inference_config: Optional[VideoInferenceConfig] = None
        self._dataset_config: Optional[DatasetConfig] = None
        self._model: Optional[nn.Module] = None

    def with_model_config(self, config: ModelConfig) -> VideoPipelineBuilder:
        """Set model configuration."""
        self._model_config = config
        return self

    def with_training_config(self, config: VideoTrainingConfig) -> VideoPipelineBuilder:
        """Set training configuration."""
        self._training_config = config
        return self

    def with_inference_config(self, config: VideoInferenceConfig) -> VideoPipelineBuilder:
        """Set inference configuration."""
        self._inference_config = config
        return self

    def with_dataset_config(self, config: DatasetConfig) -> VideoPipelineBuilder:
        """Set dataset configuration."""
        self._dataset_config = config
        return self

    def with_model(self, model: nn.Module) -> VideoPipelineBuilder:
        """Provide custom PyTorch model module."""
        self._model = model
        return self

    def build_training_pipeline(self) -> TrainingPipeline:
        """Construct TrainingPipeline instance."""
        m_cfg = self._model_config or ModelConfig()
        t_cfg = self._training_config or VideoTrainingConfig()
        model = self._model or VideoFactory.create_model(m_cfg)

        return TrainingPipeline(
            training_config=t_cfg,
            model_config=m_cfg,
            model=model,
        )

    def build_inference_pipeline(self) -> InferencePipeline:
        """Construct InferencePipeline instance."""
        m_cfg = self._model_config or ModelConfig()
        i_cfg = self._inference_config or VideoInferenceConfig()
        model = self._model or VideoFactory.create_model(m_cfg)

        return InferencePipeline(model=model, config=i_cfg)
