"""
CLI interface for audio deepfake prediction.

Usage::

    python -m app.audio.inference.predict

Improvements over v1
─────────────────────
• Error handling for missing model file
• Path validation for file input
• Graceful Ctrl+C handling
• Uses project logger
• Displays structured PredictionResult
"""

from __future__ import annotations

import sys

import torch

from app.audio.inference.predictor import Predictor
from app.audio.inference.voice_detector import VoiceDetector
from app.audio.models.cnn_model import DeepFakeCNN
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_model() -> DeepFakeCNN:
    """
    Load the trained model from disk.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint does not exist.
    """
    model_path = settings.MODEL_SAVE_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Train a model first with: python -m app.audio.training.train"
        )

    model = DeepFakeCNN(num_classes=settings.model.num_classes).to(settings.DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=settings.DEVICE, weights_only=True)
    )
    model.eval()
    logger.info("Model loaded from %s", model_path)
    return model


def predict_from_file(predictor: Predictor) -> None:
    """Predict on an audio file provided via CLI input."""
    audio_path = input("\nEnter audio file path: ").strip()
    if not audio_path:
        print("No path provided.")
        return

    try:
        result = predictor.predict(audio_path)
        print("\n" + "=" * 40)
        print("         PREDICTION RESULT")
        print("=" * 40)
        print(f"  Label:      {result.label.value}")
        print(f"  Confidence: {result.confidence:.4f} ({result.confidence * 100:.1f}%)")
        print(f"  Latency:    {result.latency_ms:.1f} ms")
        print(f"  Modality:   {result.modality.value}")
        print("=" * 40)
    except FileNotFoundError as exc:
        print(f"\n❌ File not found: {exc}")
    except ValueError as exc:
        print(f"\n❌ Invalid file: {exc}")
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        print(f"\n❌ Prediction error: {exc}")


def predict_from_microphone(detector: VoiceDetector) -> None:
    """Record from microphone and predict."""
    try:
        result = detector.detect(duration=settings.audio.recording_duration)
        print("\n" + "=" * 40)
        print("         PREDICTION RESULT")
        print("=" * 40)
        print(f"  Label:      {result.label.value}")
        print(f"  Confidence: {result.confidence:.4f} ({result.confidence * 100:.1f}%)")
        print(f"  Latency:    {result.latency_ms:.1f} ms")
        print("=" * 40)
    except Exception as exc:
        logger.error("Microphone prediction failed: %s", exc)
        print(f"\n❌ Error: {exc}")


def main() -> None:
    """Interactive CLI for deepfake detection."""
    logger.info("Starting DeepFake Detector CLI")
    print(f"\nUsing device: {settings.DEVICE}")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    predictor = Predictor(model, settings.DEVICE)
    detector = VoiceDetector(predictor)

    try:
        while True:
            print("\n" + "=" * 40)
            print("     DeepFake Audio Detector")
            print("=" * 40)
            print("  1. Predict from audio file")
            print("  2. Record from microphone")
            print("  3. Exit")

            choice = input("\nChoose option: ").strip()

            if choice == "1":
                predict_from_file(predictor)
            elif choice == "2":
                predict_from_microphone(detector)
            elif choice == "3":
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice. Please enter 1, 2, or 3.")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()