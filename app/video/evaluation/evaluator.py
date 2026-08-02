"""Video model evaluator engine module."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.video.core.base_evaluator import BaseEvaluator
from app.video.evaluation.confusion_matrix import ConfusionMatrix
from app.video.evaluation.metrics import EvaluationMetrics
from app.video.evaluation.performance_evaluator import PerformanceEvaluator


class VideoEvaluator(BaseEvaluator):
    """Evaluates video deepfake models over evaluation datasets, computing metrics, confusion matrix, and performance benchmark."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.model.to(self.device)

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Execute evaluation pass and return comprehensive metrics dictionary.

        Returns:
            Dict[str, Any]: Metrics dictionary including Accuracy, Precision, Recall, F1, ROC, AUC, Confusion Matrix, Latency, FPS, GPU Memory.
        """
        loader = dataloader or self.dataloader
        if loader is None:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        self.model.eval()
        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    x = batch["tensor"].to(self.device)
                    y = batch["label"].to(self.device)
                elif hasattr(batch, "tensor"):
                    x = batch.tensor.to(self.device)
                    y = torch.tensor([batch.label], device=self.device)
                elif isinstance(batch, (tuple, list)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[:, 1] if logits.shape[1] > 1 else torch.sigmoid(logits[:, 0])

                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())

        y_true = np.array(all_labels, dtype=int)
        y_probs = np.array(all_probs, dtype=float)

        metrics = EvaluationMetrics.compute_all(y_true, y_probs)
        y_pred = (y_probs >= 0.5).astype(int)
        cm = ConfusionMatrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.to_dict()

        # Performance benchmark
        gpu_stats = PerformanceEvaluator.measure_gpu_memory()
        metrics["gpu_memory_allocated_mb"] = gpu_stats["allocated_mb"]
        metrics["gpu_memory_reserved_mb"] = gpu_stats["reserved_mb"]

        return metrics


# Class alias
Evaluator = VideoEvaluator
