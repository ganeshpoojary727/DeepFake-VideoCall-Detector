"""Tests for the DeepFakeCNN model architecture."""

from __future__ import annotations

import torch
import pytest

from app.audio.models.cnn_model import DeepFakeCNN
from app.config.settings import settings


class TestDeepFakeCNN:
    """Test suite for the CNN model."""

    def test_output_shape(self, model: DeepFakeCNN, dummy_batch: torch.Tensor) -> None:
        """Model output should have shape (batch, num_classes)."""
        output = model(dummy_batch)
        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

    def test_single_sample(self, model: DeepFakeCNN) -> None:
        """Model should handle a single sample."""
        x = torch.randn(1, 1, settings.audio.n_mels, settings.audio.target_length)
        output = model(x)
        assert output.shape == (1, 2)

    def test_variable_time_frames(self, model: DeepFakeCNN) -> None:
        """AdaptiveAvgPool2d should handle variable time dimensions."""
        for length in [50, 100, 200, 300]:
            x = torch.randn(1, 1, settings.audio.n_mels, length)
            output = model(x)
            assert output.shape == (1, 2), (
                f"Failed for time_frames={length}: got {output.shape}"
            )

    def test_variable_mel_bins(self, model: DeepFakeCNN) -> None:
        """AdaptiveAvgPool2d should handle variable mel dimensions."""
        for n_mels in [64, 128, 256]:
            x = torch.randn(1, 1, n_mels, 100)
            output = model(x)
            assert output.shape == (1, 2), (
                f"Failed for n_mels={n_mels}: got {output.shape}"
            )

    def test_custom_num_classes(self) -> None:
        """Model should support custom number of output classes."""
        model = DeepFakeCNN(num_classes=5)
        x = torch.randn(2, 1, 128, 100)
        output = model(x)
        assert output.shape == (2, 5)

    def test_parameter_count(self, model: DeepFakeCNN) -> None:
        """Model should be lightweight (~67K params, not 1.6M)."""
        total = sum(p.numel() for p in model.parameters())
        assert total < 100_000, (
            f"Model has {total:,} params — expected <100K after AdaptiveAvgPool2d fix"
        )

    def test_gradient_flow(self, model: DeepFakeCNN) -> None:
        """Gradients should flow through the entire model."""
        x = torch.randn(2, 1, 128, 100, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "No gradients flowed back to input"

    def test_eval_mode(self, model: DeepFakeCNN) -> None:
        """Model should produce deterministic output in eval mode."""
        model.eval()
        x = torch.randn(1, 1, 128, 100)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2), "Eval mode should be deterministic"