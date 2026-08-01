"""
Audio preprocessing and feature extraction pipeline module.

Provides the AudioPreprocessingPipeline class for chaining audio loading, trimming,
normalization, Mel-spectrogram feature extraction, and data augmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from app.audio.augmentation.augmentation import AugmentationPipeline
from app.audio.configs.dataset_config import AudioFeatureConfig
from app.audio.features.feature_extractor import FeatureExtractor
from app.audio.preprocessing.audio_preprocessor import AudioPreprocessor
from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("preprocessing.pipeline")


class AudioPreprocessingPipeline:
    """
    Unified end-to-end preprocessing pipeline for raw audio waveform inputs.

    Parameters
    ----------
    config : Optional[AudioFeatureConfig]
        Feature extraction configuration parameters.
    augmentation_pipeline : Optional[AugmentationPipeline]
        Data augmentation pipeline for training data.
    """

    def __init__(
        self,
        config: Optional[AudioFeatureConfig] = None,
        augmentation_pipeline: Optional[AugmentationPipeline] = None,
    ) -> None:
        self.config = config or AudioFeatureConfig()
        self.preprocessor = AudioPreprocessor(sample_rate=self.config.sample_rate)
        self.feature_extractor = FeatureExtractor(
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            target_length=self.config.target_length,
        )
        self.augmentation_pipeline = augmentation_pipeline

    def process_file(
        self,
        audio_path: Union[str, Path],
        apply_augmentation: bool = False,
    ) -> torch.Tensor:
        """
        Load audio file from disk and extract normalised feature tensor.

        Parameters
        ----------
        audio_path : Union[str, Path]
            Path to raw audio file (.wav, .flac, .mp3, etc.).
        apply_augmentation : bool
            Whether to apply augmentation transformations.

        Returns
        -------
        torch.Tensor
            Extracted Mel-spectrogram tensor of shape (1, n_mels, target_length).
        """
        path = Path(audio_path)
        audio, sr = self.preprocessor.preprocess(path)

        if apply_augmentation and self.augmentation_pipeline is not None:
            audio = self.augmentation_pipeline(audio, sr=sr)

        feature_tensor = self.feature_extractor.extract(audio)
        return feature_tensor

    def process_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        apply_augmentation: bool = False,
    ) -> torch.Tensor:
        """
        Process raw audio waveform array into normalised feature tensor.

        Parameters
        ----------
        audio_array : np.ndarray
            Raw audio samples array.
        sample_rate : int
            Sampling rate of the input array.
        apply_augmentation : bool
            Whether to apply data augmentation.

        Returns
        -------
        torch.Tensor
            Extracted Mel-spectrogram tensor of shape (1, n_mels, target_length).
        """
        if sample_rate != self.config.sample_rate:
            audio = self.preprocessor.resample(audio_array, orig_sr=sample_rate)
        else:
            audio = audio_array

        audio = self.preprocessor.trim_silence(audio)
        audio = self.preprocessor.normalize_audio(audio)

        if apply_augmentation and self.augmentation_pipeline is not None:
            audio = self.augmentation_pipeline(audio, sr=self.config.sample_rate)

        feature_tensor = self.feature_extractor.extract(audio)
        return feature_tensor
