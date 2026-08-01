"""
Model loading utilities with error handling and integrity verification.

Responsibilities
────────────────
• Load a ``DeepFakeCNN`` checkpoint from disk
• Verify model file exists before loading
• Handle device mapping (CPU ↔ GPU)
• Optional SHA-256 integrity check
• Warm-up inference to pre-allocate CUDA memory
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from app.audio.models.cnn_model import DeepFakeCNN
from app.config.settings import settings
from app.utils.helpers import verify_model_integrity
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Factory that loads, verifies, and prepares a ``DeepFakeCNN`` for inference.

    Parameters
    ----------
    model_path : Path | str | None
        Path to the ``.pth`` checkpoint.  Defaults to ``settings.MODEL_SAVE_PATH``.
    device : torch.device | None
        Target device.  Defaults to ``settings.DEVICE``.
    expected_hash : str | None
        If provided, the checkpoint is verified against this SHA-256 digest
        before loading.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
        expected_hash: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path or settings.MODEL_SAVE_PATH)
        self.device = device or settings.DEVICE
        self.expected_hash = expected_hash

    def load(self, warmup: bool = True) -> DeepFakeCNN:
        """
        Load and return the model, ready for inference.

        Parameters
        ----------
        warmup : bool
            If ``True``, run a dummy forward pass to pre-allocate GPU memory.

        Returns
        -------
        DeepFakeCNN
            Model in ``eval()`` mode on the target device.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        RuntimeError
            If integrity verification fails or weights cannot be loaded.
        """
        # ── Existence check ───────────────────
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        # ── Integrity check ───────────────────
        if self.expected_hash is not None:
            if not verify_model_integrity(self.model_path, self.expected_hash):
                raise RuntimeError(
                    "Model integrity verification failed — "
                    "the checkpoint may have been tampered with."
                )
            logger.info("Model integrity verified ✓")

        # ── Load ──────────────────────────────
        logger.info("Loading model (%s) from %s", settings.model.model_name, self.model_path)
        from app.audio.models.model_registry import ModelRegistry
        model = ModelRegistry.create(
            settings.model.model_name,
            num_classes=settings.model.num_classes,
        )
        state_dict = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        logger.info(
            "Model loaded on %s  (params: %s)",
            self.device,
            f"{sum(p.numel() for p in model.parameters()):,}",
        )

        # ── Warm-up ──────────────────────────
        if warmup:
            self._warmup(model)

        return model

    def _warmup(self, model: DeepFakeCNN) -> None:
        """Run a dummy forward pass to trigger CUDA kernel compilation."""
        try:
            dummy = torch.randn(
                1, 1,
                settings.audio.n_mels,
                settings.audio.target_length,
                device=self.device,
            )
            with torch.no_grad():
                model(dummy)
            logger.debug("Warm-up inference complete")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warm-up inference failed: %s", exc)
