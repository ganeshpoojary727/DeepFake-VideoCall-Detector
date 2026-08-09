"""Validation engine module for evaluating audio AASIST models."""

from __future__ import annotations

import math
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

    def __init__(self, model: nn.Module, device: str = "cpu", use_amp: Optional[bool] = None) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        if use_amp is not None:
            self.use_amp = bool(use_amp) and self.device.type == "cuda"
        else:
            self.use_amp = self.device.type == "cuda"

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

        valid_loss_samples = 0
        has_non_finite = False
        use_amp = self.use_amp

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

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    has_non_finite = True

                if criterion is not None:
                    # Always evaluate criterion in FP32 precision
                    loss = criterion(logits.float(), y)
                    loss_val = loss.item()
                    if torch.isnan(loss) or torch.isinf(loss) or math.isnan(loss_val) or math.isinf(loss_val):
                        has_non_finite = True
                        logger.warning("Non-finite validation loss encountered in batch evaluation.")
                    else:
                        total_loss += loss_val * x.size(0)
                        valid_loss_samples += x.size(0)

                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        val_loss = (total_loss / valid_loss_samples) if (valid_loss_samples > 0 and not has_non_finite) else None
        avg_latency = total_time_ms / sample_count if sample_count > 0 else 0.0

        if all_logits:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            metrics = AudioMetricsCalculator.compute_all(cat_logits, cat_labels, latency_ms=avg_latency)
        else:
            metrics = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "confusion_matrix": None,
                "latency_ms": 0.0,
                "gpu_memory_mb": 0.0,
                "eer": None,
                "hter": None,
                "apcer": None,
                "bpcer": None,
                "eer_threshold": None,
                "is_valid": False,
            }

        val_time_sec = time.perf_counter() - val_start_time
        metrics["val_loss"] = float(val_loss) if val_loss is not None else None
        metrics["val_time_sec"] = float(val_time_sec)
        if has_non_finite:
            metrics["is_valid"] = False
        return metrics
