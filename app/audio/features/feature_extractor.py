"""
Feature extraction module — converts preprocessed audio to CNN-ready tensors.

Pipeline
────────
mel spectrogram → dB conversion → per-sample normalisation → pad / crop → tensor

Improvements over v1
─────────────────────
• Reads hyper-parameters from ``Settings`` (fixes parameter disconnection)
• Per-sample z-normalisation (audit §5.4)
• Optional SpecAugment (frequency + time masking) for training
• Uses ``torch.from_numpy`` instead of ``torch.tensor`` (zero-copy)
• Consistent 4-space indentation throughout
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np
import torch

from app.config.settings import settings
from app.core.interfaces import BaseFeatureExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor(BaseFeatureExtractor):
    """
    Extract Mel-spectrogram features from preprocessed audio waveforms.

    Parameters
    ----------
    sample_rate : int | None
        Defaults to ``settings.audio.sample_rate``.
    n_fft : int | None
        Defaults to ``settings.audio.n_fft``.
    hop_length : int | None
        Defaults to ``settings.audio.hop_length``.
    n_mels : int | None
        Defaults to ``settings.audio.n_mels``.
    target_length : int | None
        Defaults to ``settings.audio.target_length``.
    apply_augmentation : bool
        If ``True``, apply SpecAugment during extraction (training only).
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: Optional[int] = None,
        target_length: Optional[int] = None,
        apply_augmentation: bool = False,
    ) -> None:
        self.sample_rate = sample_rate or settings.audio.sample_rate
        self.n_fft = n_fft or settings.audio.n_fft
        self.hop_length = hop_length or settings.audio.hop_length
        self.n_mels = n_mels or settings.audio.n_mels
        self.target_length = target_length or settings.audio.target_length
        self.apply_augmentation = apply_augmentation

    # ── Public API (satisfies BaseFeatureExtractor) ──

    def extract(self, audio: np.ndarray) -> torch.Tensor:
        """
        Full feature extraction pipeline.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 waveform.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(1, n_mels, target_length)``.
        """
        mel = self.create_mel_spectrogram(audio)
        mel_db = self.convert_to_db(mel)
        mel_db = self.normalize_spectrogram(mel_db)
        mel_db = self.resize_spectrogram(mel_db)
        tensor = self.to_tensor(mel_db)

        if self.apply_augmentation:
            tensor = self.spec_augment(tensor)

        return tensor

    # ── Individual steps ──────────────────────

    def create_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Generate a Mel spectrogram from an audio waveform."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return mel

    def convert_to_db(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a power spectrogram to decibel scale."""
        return librosa.power_to_db(mel_spectrogram, ref=np.max)

    def normalize_spectrogram(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Per-sample zero-mean unit-variance normalisation.

        This is critical for CNN convergence — raw dB values range
        from 0 to −80 which forces the network to learn the scale.
        """
        mean = mel_db.mean()
        std = mel_db.std()
        return (mel_db - mean) / (std + 1e-9)

    def resize_spectrogram(self, mel_db: np.ndarray) -> np.ndarray:
        """Pad or crop the spectrogram to ``target_length`` time frames."""
        current_length = mel_db.shape[1]

        if current_length < self.target_length:
            pad_width = self.target_length - current_length
            mel_db = np.pad(
                mel_db,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant",
            )
        else:
            mel_db = mel_db[:, : self.target_length]

        return mel_db

    def to_tensor(self, mel_db: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy spectrogram to a PyTorch tensor.

        Uses ``torch.from_numpy`` for zero-copy conversion (audit §2.4).

        Returns
        -------
        torch.Tensor
            Shape ``(1, n_mels, target_length)`` — the channel dim is added.
        """
        tensor = torch.from_numpy(mel_db.copy()).float()
        tensor = tensor.unsqueeze(0)  # add channel dim
        return tensor

    def spec_augment(
        self,
        tensor: torch.Tensor,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
    ) -> torch.Tensor:
        """
        Apply SpecAugment (frequency and time masking) for regularisation.

        Only active when ``self.apply_augmentation`` is ``True``.
        """
        try:
            import torchaudio.transforms as T

            freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            time_mask = T.TimeMasking(time_mask_param=time_mask_param)
            tensor = freq_mask(tensor)
            tensor = time_mask(tensor)
        except ImportError:
            logger.warning(
                "torchaudio not available — SpecAugment skipped. "
                "Install torchaudio for data augmentation."
            )
        return tensor