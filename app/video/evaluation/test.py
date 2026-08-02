"""Standalone evaluation execution script for video deepfake models."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.video.evaluation.evaluator import VideoEvaluator


def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute complete evaluation pass on test dataloader.

    Args:
        model: PyTorch model.
        test_loader: Test set dataloader.
        device: Computing device string.

    Returns:
        Dict[str, Any]: Evaluated metrics dictionary.
    """
    evaluator = VideoEvaluator(model=model, dataloader=test_loader, device=device)
    return evaluator.evaluate()
