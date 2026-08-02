"""CLI interface for audio deepfake prediction using production AASIST model."""

from __future__ import annotations

import sys
import torch

from app.audio.inference.predictor import Predictor
from app.audio.inference.voice_detector import VoiceDetector
from app.audio.models.aasist import AASIST
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_model() -> AASIST:
    """Load the trained AASIST model from disk."""
    model_path = settings.MODEL_SAVE_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Train a model first with: python -m app.audio.training.train"
        )

    model = AASIST(num_classes=settings.model.num_classes).to(settings.DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=settings.DEVICE, weights_only=True)
    )
    model.eval()
    logger.info("AASIST Model loaded from %s", model_path)
    return model


def predict_from_file(predictor: Predictor) -> None:
    """Predict on an audio file provided via CLI input."""
    audio_path = input("\nEnter audio file path: ").strip()
    if not audio_path:
        return
    res = predictor.predict_file(audio_path)
    print(f"Prediction: {res}")


def main() -> None:
    """Main CLI runner."""
    model = load_model()
    detector = VoiceDetector()
    detector._model = model
    print("AASIST Voice Detector CLI Ready.")


if __name__ == "__main__":
    main()