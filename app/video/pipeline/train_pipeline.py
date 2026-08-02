"""Training pipeline entrypoint script for video subsystem."""

from __future__ import annotations

from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.pipeline.training_pipeline import TrainingPipeline


def run_training_pipeline(
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[VideoTrainingConfig] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Execute video model training pipeline.

    Returns:
        Dict[str, Any]: History metrics dictionary.
    """
    pipeline = TrainingPipeline(
        model_config=model_config,
        training_config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    return pipeline.execute()
