"""
Application entry point.

Default behavior: launches the PyQt6 GUI directly.
CLI subcommands are still available for training, evaluation, etc.

Usage::

    python app/main.py              # Launch GUI (default)
    python -m app.main              # Launch GUI (default)
    python -m app.main train        # Train model
    python -m app.main predict      # Interactive prediction CLI
    python -m app.main evaluate     # Run evaluation on test set
    python -m app.main info         # Show project info
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _show_info() -> None:
    """Display project information."""
    warnings = settings.validate()

    print("\n" + "=" * 60)
    print("  DeepFake Video Call Detector v4.0")
    print("=" * 60)
    print(f"  Device:           {settings.DEVICE}")
    print(f"  Model:            {settings.model.model_name} v{settings.model.model_version}")
    print(f"  Model path:       {settings.MODEL_SAVE_PATH}")
    print(f"  Model exists:     {settings.MODEL_SAVE_PATH.exists()}")
    print(f"  Sample rate:      {settings.audio.sample_rate} Hz")
    print(f"  n_mels:           {settings.audio.n_mels}")
    print(f"  Warmup duration:  {settings.INITIAL_WARMUP_SEC}s")
    print(f"  Chunk interval:   {settings.INCREMENTAL_CHUNK_SEC}s")
    print(f"  Video FPS:        {settings.VIDEO_TARGET_FPS}")
    print(f"  Audio weight:     {settings.AUDIO_WEIGHT:.0%}")
    print(f"  Video weight:     {settings.VIDEO_WEIGHT:.0%}")
    print(f"  Batch size:       {settings.training.batch_size}")
    print(f"  Learning rate:    {settings.training.learning_rate}")
    print(f"  Epochs:           {settings.training.epochs}")

    if warnings:
        print("\n⚠️  Configuration warnings:")
        for w in warnings:
            print(f"    • {w}")

    print("=" * 60)


def _launch_gui() -> None:
    """Launch the PyQt6 GUI application."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except ImportError:
        logger.error("PyQt6 is not installed")
        print("❌ PyQt6 required. Install with: pip install PyQt6")
        sys.exit(1)

    # ── High-DPI Scaling ──────────────────────
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("DeepFake Video Call Detector")
    app.setApplicationVersion("4.0")

    from app.gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    logger.info("GUI launched — entering event loop")
    sys.exit(app.exec())


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate module."""
    parser = argparse.ArgumentParser(
        prog="deepfake-detector",
        description="DeepFake Video Call Detector — AI-powered deepfake detection",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Subcommands ───────────────────────────
    subparsers.add_parser("train", help="Train the audio deepfake detection model")
    subparsers.add_parser("train-video", help="Train the video deepfake detection model")
    subparsers.add_parser("predict", help="Interactive prediction CLI")
    subparsers.add_parser("evaluate", help="Evaluate model on test set")
    subparsers.add_parser("gui", help="Launch the GUI application")
    subparsers.add_parser("info", help="Show project configuration info")

    args, _ = parser.parse_known_args()

    # ── Default: launch GUI if no subcommand ──
    if args.command is None or args.command == "gui":
        _launch_gui()
        return

    logger.info("DeepFake Video Call Detector — mode: %s", args.command)

    if args.command == "train":
        from app.audio.training.train import main as train_main
        train_main()

    elif args.command == "train-video":
        from app.video.training.train_video import main as train_video_main
        train_video_main()

    elif args.command == "predict":
        from app.audio.inference.predict import main as predict_main
        predict_main()

    elif args.command == "evaluate":
        from app.audio.evaluation.test import main as evaluate_main
        evaluate_main()

    elif args.command == "info":
        _show_info()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()