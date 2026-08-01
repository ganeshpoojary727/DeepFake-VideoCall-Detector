"""Video model validation pipeline module."""

from __future__ import annotations

from typing import Dict, Optional
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.video.datasets.video_sample import video_collate_fn
from app.video.training.metrics import VideoMetricsCalculator
from app.video.training.trainer import Trainer
from app.video.utils.device import get_device


class ValidationPipeline:
    """Evaluates video deepfake models across validation datasets."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)

    def evaluate(self, val_dataset: Dataset, batch_size: int = 4) -> Dict[str, float]:
        """Run complete validation evaluation.

        Args:
            val_dataset: Target dataset to evaluate.
            batch_size: DataLoader batch size.

        Returns:
            Dict[str, float]: Validation metric scores dictionary.
        """
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=video_collate_fn,
        )
        trainer = Trainer(model=self.model)
        return trainer.validate(val_loader)
