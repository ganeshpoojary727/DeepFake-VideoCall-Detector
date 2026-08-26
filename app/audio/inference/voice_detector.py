"""Voice Detector — in-memory buffer-based audio deepfake detection using AASIST."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from app.audio.models.aasist import AASIST
from app.audio.registry.model_registry import ModelRegistry
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceDetector:
    """Audio deepfake detector operating on in-memory buffers using AASIST model."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or settings.DEVICE
        self._model: Optional[torch.nn.Module] = None
        self._model_loaded = False

        model_path = Path(model_path or settings.MODEL_SAVE_PATH)
        if model_path.exists():
            try:
                self._load_model(model_path)
            except Exception as exc:
                logger.warning("VoiceDetector: model load failed: %s", exc)
        else:
            logger.info("VoiceDetector: no checkpoint at %s — initializing default AASIST", model_path)
            self._model = AASIST(num_classes=settings.model.num_classes).to(self.device)
            self._model.eval()
            self._model_loaded = True

    def _load_model(self, path: Path) -> None:
        """Load AASIST model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model = AASIST(num_classes=settings.model.num_classes)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        self._model = model
        self._model_loaded = True
        logger.info("VoiceDetector: loaded AASIST model from %s", path)

    def predict_buffer(self, audio_buffer: np.ndarray) -> float:
        """Predict fake probability from in-memory audio numpy buffer.

        Args:
            audio_buffer: Audio sample array.

        Returns:
            float: Deepfake probability score in [0.0, 1.0].
        """
        if self._model is None:
            return 0.5

        tensor = torch.from_numpy(audio_buffer).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._model(tensor)
            if isinstance(out, tuple):
                logits = out[1] if len(out) > 1 else out[0]
            else:
                logits = out
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)