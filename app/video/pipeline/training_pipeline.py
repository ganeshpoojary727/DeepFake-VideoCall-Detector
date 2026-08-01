"""Video model training pipeline orchestration module."""

from __future__ import annotations

from typing import Any, Dict, Optional
from torch.utils.data import DataLoader, Dataset

from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.datasets.video_sample import video_collate_fn
from app.video.models.base_video_model import BaseVideoModel
from app.video.models.video_factory import VideoFactory
from app.video.training.trainer import Trainer


class TrainingPipeline:
    """Orchestrates model creation, dataset loading, and trainer execution."""

    def __init__(
        self,
        training_config: Optional[VideoTrainingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        model: Optional[BaseVideoModel] = None,
    ) -> None:
        self.training_config = training_config or VideoTrainingConfig()
        self.model_config = model_config or ModelConfig()
        self.model = model or VideoFactory.create_model(self.model_config)
        self.trainer = Trainer(model=self.model, config=self.training_config)

    def run(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Execute training and validation pipeline steps.

        Args:
            train_dataset: PyTorch dataset for model training.
            val_dataset: Optional dataset for validation evaluation.

        Returns:
            Dict[str, Any]: History dictionary of loss and metrics.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=video_collate_fn,
        )

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.training_config.epochs):
            train_loss = self.trainer.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.training_config.batch_size,
                    collate_fn=video_collate_fn,
                )
                val_metrics = self.trainer.validate(val_loader)
                val_loss = val_metrics.get("val_loss", 0.0)
                history["val_loss"].append(val_loss)

                if self.trainer.early_stopping(val_loss):
                    break

        return history
