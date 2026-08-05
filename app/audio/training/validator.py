"""Validation engine module for evaluating audio AASIST models."""

from __future__ import annotations

import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.audio.training.metrics import AudioMetricsCalculator
from app.audio.utils.audio_logger import get_audio_logger

logger = get_audio_logger("training.validator")


class ValidationEngine:
    """Evaluates AASIST models across validation datasets and computes comprehensive metrics."""

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(self, val_loader: DataLoader, criterion: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Run complete validation evaluation pass.

        Args:
            val_loader: DataLoader for validation split.
            criterion: Optional PyTorch loss module.

        Returns:
            Dict[str, Any]: Comprehensive metrics dictionary.
        """
        self.model.eval()
        val_start_time = time.perf_counter()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        total_time_ms = 0.0
        sample_count = 0

        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch["tensor"].to(self.device, non_blocking=True)
                    y = batch["label"].to(self.device, non_blocking=True)
                elif isinstance(batch, (tuple, list)):
                    x = batch[0].to(self.device, non_blocking=True)
                    y = batch[1].to(self.device, non_blocking=True)
                else:
                    continue

                t0 = time.perf_counter()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(x)
                else:
                    logits = self.model(x)
                t1 = time.perf_counter()

                total_time_ms += (t1 - t0) * 1000.0
                sample_count += x.size(0)

                if criterion is not None:
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            loss = criterion(logits, y)
                    else:
                        loss = criterion(logits, y)
                    total_loss += loss.item() * x.size(0)

                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        val_loss = total_loss / sample_count if sample_count > 0 else 0.0
        avg_latency = total_time_ms / sample_count if sample_count > 0 else 0.0

        if all_logits:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            metrics = AudioMetricsCalculator.compute_all(cat_logits, cat_labels, latency_ms=avg_latency)
        else:
            metrics = {"accuracy": 0.0, "f1": 0.0, "eer": 0.0}

        val_time_sec = time.perf_counter() - val_start_time
        metrics["val_loss"] = float(val_loss)
        metrics["val_time_sec"] = float(val_time_sec)
        return metrics
