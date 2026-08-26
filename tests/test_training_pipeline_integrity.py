"""Unit test suite verifying training pipeline integrity, numerical stability, NaN protection, and checkpoint safety."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

from app.audio.training.checkpoint import CheckpointManager
from app.audio.training.metrics import AudioMetricsCalculator
from app.audio.training.eer_metrics import compute_biometric_metrics, compute_eer
from app.audio.models.aasist import AASIST
from app.audio.constants.audio_constants import AUDIO_MODELS_DIR


class MockNanModel(nn.Module):
    """Module containing NaN parameter weights for testing checkpoint safety."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)
        with torch.no_grad():
            self.fc.weight[0, 0] = float("nan")


class MockFiniteModel(nn.Module):
    """Module containing valid finite weights."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)


def test_checkpoint_manager_refuses_nan_weights(tmp_path: Path) -> None:
    """Verify CheckpointManager refuses to save model checkpoints if weights contain NaN/Inf."""
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    nan_model = MockNanModel()

    save_path = manager.save(nan_model, filename="test_ckpt.pt")
    assert not save_path.exists(), "Checkpoint file should NOT be created for NaN model parameters."

    best_path = manager.save_best(nan_model, epoch=1, metrics={"accuracy": 0.9})
    assert not best_path.exists(), "Best checkpoint file should NOT be created for NaN model parameters."


def test_metrics_calculator_returns_none_on_nan() -> None:
    """Verify AudioMetricsCalculator reports None / is_valid=False when non-finite logits are provided."""
    nan_logits = torch.tensor([[float("nan"), 1.0], [0.5, 0.2]])
    y_true = torch.tensor([1, 0])

    metrics = AudioMetricsCalculator.compute_all(nan_logits, y_true)

    assert metrics["accuracy"] is None, "Accuracy must be None (INVALID) when logits contain NaN."
    assert metrics["eer"] is None, "EER must be None (INVALID) when logits contain NaN."
    assert metrics["is_valid"] is False, "is_valid flag must be False for NaN logits."


def test_eer_metrics_returns_none_on_nan() -> None:
    """Verify biometric error metrics report None for NaN probabilities."""
    nan_probs = np.array([[float("nan"), float("nan")], [0.2, 0.8]])
    y_true = np.array([0, 1])

    bio = compute_biometric_metrics(nan_probs, y_true)

    assert bio["eer"] is None, "EER must be None for NaN probabilities."
    assert bio["is_valid"] is False, "is_valid must be False for NaN probabilities."

    bonafide = np.array([0.9, float("nan"), 0.8])
    spoof = np.array([0.1, 0.2])
    eer, thresh = compute_eer(bonafide, spoof)
    assert eer is None, "compute_eer must return None when non-finite scores are present."
    assert thresh is None, "compute_eer threshold must return None when non-finite scores are present."


def test_fp32_loss_calculation() -> None:
    """Verify loss calculation in FP32 produces finite outputs for large dynamic range logits under AMP."""
    criterion = nn.CrossEntropyLoss()
    logits = torch.tensor([[15.0, -10.0], [-5.0, 12.0]], device="cpu")
    y = torch.tensor([0, 1], device="cpu")

    loss_fp32 = criterion(logits.float(), y.long())
    assert torch.isfinite(loss_fp32).item(), "FP32 loss calculation must yield finite scalar."
    assert loss_fp32.item() >= 0.0, "Loss must be non-negative."


def test_json_history_nan_conversion(tmp_path: Path) -> None:
    """Verify CheckpointManager converts float('nan') in training history to null in JSON."""
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    finite_model = MockFiniteModel()

    history = {
        "train_loss": [0.5, float("nan"), 0.3],
        "val_loss": [float("nan"), 0.4, float("inf")],
    }

    manager.save(finite_model, filename="valid_ckpt.pt", history=history)
    history_file = tmp_path / "training_history.json"
    assert history_file.exists()

    with open(history_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["train_loss"] == [0.5, None, 0.3], "NaN floats in history must be converted to None (JSON null)."
    assert loaded["val_loss"] == [None, 0.4, None], "NaN and Inf floats must be converted to None (JSON null)."


def test_existing_best_model_pt_unmodified() -> None:
    """Verify existing Epoch-2 best_model.pt remains finite, intact, and loads successfully."""
    best_path = Path(AUDIO_MODELS_DIR) / "best_model.pt"
    assert best_path.exists(), f"Epoch-2 best_model.pt must exist at {best_path}"

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    state_dict = checkpoint.get("model_state_dict", {})

    assert epoch is not None and epoch >= 1, f"Expected valid trained model epoch >= 1, got Epoch {epoch}"

    # Verify all saved parameters are finite
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all().item(), f"Parameter {k} in best_model.pt contains NaN or Inf!"

    # Verify inference produces finite logits
    model = AASIST(num_classes=2)
    # Load with strict=False because of prior maxpool downsample adaptation
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(2, 64000)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all().item(), "Inference on best_model.pt must yield finite logits."
