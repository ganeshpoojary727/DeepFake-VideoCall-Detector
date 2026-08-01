"""
Spectrogram visualisation utility.

Usage::

    python -m app.tools.visualize_spectrogram path/to/audio.flac

Generates and displays a Mel spectrogram for a given audio file,
useful for debugging feature extraction and verifying preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.audio.preprocessing.audio_preprocessor import AudioPreprocessor
from app.audio.features.feature_extractor import FeatureExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)


def visualize(audio_path: str | Path, save_path: str | None = None) -> None:
    """
    Generate and display a Mel spectrogram visualisation.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file.
    save_path : str | None
        If provided, save the figure to this path instead of displaying.
    """
    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()

    audio, sr = preprocessor.preprocess(audio_path)
    mel = extractor.create_mel_spectrogram(audio)
    mel_db = extractor.convert_to_db(mel)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Waveform
    time_axis = np.arange(len(audio)) / sr
    axes[0].plot(time_axis, audio, linewidth=0.5, color="#89b4fa")
    axes[0].set_title("Waveform", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim([0, time_axis[-1]])

    # Mel spectrogram
    img = axes[1].imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, mel_db.shape[1], 0, mel_db.shape[0]],
    )
    axes[1].set_title("Mel Spectrogram (dB)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Bins")
    fig.colorbar(img, ax=axes[1], label="dB")

    plt.suptitle(
        f"Audio Analysis: {Path(audio_path).name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    else:
        plt.show()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualise Mel Spectrograms")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--save", help="Save figure to path instead of displaying")
    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        print(f"❌ File not found: {args.audio_path}")
        sys.exit(1)

    visualize(args.audio_path, save_path=args.save)


if __name__ == "__main__":
    main()
