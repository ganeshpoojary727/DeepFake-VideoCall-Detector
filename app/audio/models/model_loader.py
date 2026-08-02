"""Model loading utilities with error handling and integrity verification.

Responsibilities
────────────────
• Load an AASIST model checkpoint from disk
• Verify model file exists before loading
• Handle device mapping (CPU ↔ GPU)
• Optional SHA-256 integrity check
• Warm-up inference to pre-allocate CUDA memory
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

from app.audio.models.aasist import AASIST
from app.audio.registry.model_registry import ModelRegistry
from app.config.settings import settings
from app.utils.helpers import verify_model_integrity
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Factory that loads, verifies, and prepares an AASIST model for inference."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
        expected_hash: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path or settings.MODEL_SAVE_PATH)
        self.device = device or settings.DEVICE
        self.expected_hash = expected_hash

    def load(self, warmup: bool = True) -> nn.Module:
        """Load and return the model, ready for inference."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        if self.expected_hash is not None:
            if not verify_model_integrity(self.model_path, self.expected_hash):
                raise RuntimeError("Model integrity verification failed")
            logger.info("Model integrity verified ✓")

        logger.info("Loading model (%s) from %s", settings.model.model_name, self.model_path)
        model = ModelRegistry.create(
            settings.model.model_name,
            num_classes=settings.model.num_classes,
        )
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        logger.info("Model loaded on %s", self.device)
        if warmup:
            self._warmup(model)

        return model

    def _warmup(self, model: nn.Module) -> None:
        """Run a dummy forward pass to trigger CUDA kernel compilation."""
        try:
            dummy = torch.randn(
                1, settings.audio.target_length,
                device=self.device,
            )
            with torch.no_grad():
                model(dummy)
            logger.debug("Warm-up inference complete")
        except Exception as exc:
            logger.warning("Warm-up inference failed: %s", exc)
