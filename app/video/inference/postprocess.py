"""Inference result postprocessing and confidence score smoothing module."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import torch


class InferencePostProcessor:
    """Post-processes raw model logits/probabilities into structured confidence outputs."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def process_outputs(
        self, logits_or_probs: torch.Tensor | np.ndarray
    ) -> Dict[str, Any]:
        """Convert raw tensor logits/probabilities into probability dict.

        Args:
            logits_or_probs: PyTorch tensor or numpy array.

        Returns:
            Dict[str, Any]: Dictionary containing probability, label, is_fake, and confidence score.
        """
        if isinstance(logits_or_probs, torch.Tensor):
            arr = logits_or_probs.detach().cpu().numpy()
        else:
            arr = np.array(logits_or_probs)

        if arr.ndim == 2:
            if arr.shape[1] == 2:
                # Logits softmax or prob array
                probs = np.exp(arr) / np.sum(np.exp(arr), axis=1, keepdims=True)
                fake_prob = float(probs[0, 1])
            else:
                fake_prob = float(arr[0, 0])
        elif arr.ndim == 1:
            fake_prob = float(arr[1]) if len(arr) > 1 else float(arr[0])
        else:
            fake_prob = float(arr)

        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
        label = 1 if fake_prob >= self.confidence_threshold else 0
        label_str = "fake" if label == 1 else "real"
        is_fake = bool(label == 1)

        return {
            "is_fake": is_fake,
            "is_deepfake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "confidence": float(abs(fake_prob - 0.5) * 2.0),
            "label": label,
            "label_name": label_str,
        }
