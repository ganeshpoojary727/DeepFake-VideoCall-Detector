"""Audio data augmentation transforms for training."""

from __future__ import annotations

import random
from typing import Callable, List
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

WaveformTransform = Callable[[np.ndarray, int], np.ndarray]


class GaussianNoise:
    """Add white Gaussian noise to the waveform."""

    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 40.0) -> None:
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        signal_power = np.mean(waveform ** 2)
        if signal_power < 1e-10:
            return waveform
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape).astype(np.float32)
        return np.clip(waveform + noise, -1.0, 1.0)


class VolumePerturbation:
    """Randomly scale the amplitude of the waveform."""

    def __init__(self, min_gain: float = 0.5, max_gain: float = 1.5) -> None:
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        gain = random.uniform(self.min_gain, self.max_gain)
        return np.clip(waveform * gain, -1.0, 1.0)


class SpeedPerturbation:
    """Simulate speed change via resampling."""

    def __init__(self, min_rate: float = 0.9, max_rate: float = 1.1) -> None:
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        try:
            import librosa
        except ImportError:
            return waveform

        rate = random.uniform(self.min_rate, self.max_rate)
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        orig_len = len(waveform)
        if len(stretched) > orig_len:
            return stretched[:orig_len]
        return np.pad(stretched, (0, orig_len - len(stretched)))


class RoomSimulation:
    """Simulate room reverb via convolution with a synthetic impulse response."""

    def __init__(self, min_rt60_ms: float = 50.0, max_rt60_ms: float = 500.0) -> None:
        self.min_rt60_ms = min_rt60_ms
        self.max_rt60_ms = max_rt60_ms

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        rt60_ms = random.uniform(self.min_rt60_ms, self.max_rt60_ms)
        rt60_samples = int(rt60_ms * sr / 1000)

        t = np.arange(rt60_samples, dtype=np.float32)
        decay = -3.0 / rt60_samples
        rir = np.exp(decay * t) * np.random.randn(rt60_samples).astype(np.float32)
        rir /= np.abs(rir).max() + 1e-8

        reverbed = np.convolve(waveform, rir)[:len(waveform)]
        wet = random.uniform(0.1, 0.4)
        return np.clip(waveform * (1 - wet) + reverbed * wet, -1.0, 1.0).astype(np.float32)


class CodecSimulation:
    """Simulate codec compression artifacts (Opus/MP3)."""

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        try:
            bits = random.choice([8, 10, 12])
            levels = 2 ** bits
            quantized = np.round(waveform * levels / 2) / (levels / 2)
            return np.clip(quantized, -1.0, 1.0).astype(np.float32)
        except Exception:
            return waveform


class SpecAugment:
    """SpecAugment: frequency and time masking for spectrogram tensors."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spectrogram) -> "torch.Tensor":
        import torch

        spec = spectrogram.clone()
        if spec.dim() == 2:
            n_freq, n_time = spec.shape
        else:
            _, n_freq, n_time = spec.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            if self.freq_mask_param > 0 and n_freq > 0:
                f = random.randint(1, min(self.freq_mask_param, n_freq))
                f0 = random.randint(0, n_freq - f)
                if spec.dim() == 2:
                    spec[f0:f0 + f, :] = 0
                else:
                    spec[:, f0:f0 + f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            if self.time_mask_param > 0 and n_time > 0:
                t = random.randint(1, min(self.time_mask_param, n_time))
                t0 = random.randint(0, n_time - t)
                if spec.dim() == 2:
                    spec[:, t0:t0 + t] = 0
                else:
                    spec[:, :, t0:t0 + t] = 0

        return spec


class AugmentationPipeline:
    """Composable augmentation pipeline for waveforms."""

    def __init__(self, transforms: List[WaveformTransform], p: float = 0.8) -> None:
        self.transforms = transforms
        self.p = p

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.p:
            return waveform
        for transform in self.transforms:
            try:
                waveform = transform(waveform, sr)
            except Exception as exc:
                logger.warning("Augmentation %s failed: %s", type(transform).__name__, exc)
        return waveform

    @classmethod
    def default_pipeline(cls, p: float = 0.8) -> "AugmentationPipeline":
        return cls(
            transforms=[
                GaussianNoise(min_snr_db=15, max_snr_db=40),
                VolumePerturbation(min_gain=0.7, max_gain=1.3),
                RoomSimulation(min_rt60_ms=30, max_rt60_ms=300),
                CodecSimulation(),
            ],
            p=p,
        )
