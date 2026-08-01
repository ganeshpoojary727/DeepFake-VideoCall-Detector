"""Modular, independently toggleable audio augmentations module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseAudioAugmentation(nn.Module, ABC):
    """Abstract base class for audio waveform and spectrogram augmentations."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation conditionally based on probability p."""
        if not self.training or torch.rand(1).item() > self.p:
            return x
        return self.apply(x)

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation logic."""
        pass


class GaussianNoise(BaseAudioAugmentation):
    """Adds additive Gaussian noise to waveform."""

    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 30.0, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        snr_db = torch.empty(1).uniform_(self.min_snr_db, self.max_snr_db).item()
        signal_power = x.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * math.sqrt(noise_power.item() + 1e-8)
        return x + noise


class BackgroundNoise(BaseAudioAugmentation):
    """Mixes low-amplitude background noise into audio signal."""

    def __init__(self, noise_level: float = 0.005, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.noise_level = noise_level

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        bg = torch.randn_like(x) * self.noise_level
        return x + bg


class Gain(BaseAudioAugmentation):
    """Applies random volume gain scaling in dB."""

    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        gain_db = torch.empty(1).uniform_(self.min_gain_db, self.max_gain_db).item()
        factor = 10 ** (gain_db / 20)
        return x * factor


class TimeMasking(BaseAudioAugmentation):
    """Zeros out a random contiguous block of time frames."""

    def __init__(self, max_mask_length: int = 4000, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.max_mask_length = max_mask_length

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        mask_len = torch.randint(1, min(self.max_mask_length, length), (1,)).item()
        start = torch.randint(0, length - mask_len + 1, (1,)).item()
        out = x.clone()
        out[..., start : start + mask_len] = 0.0
        return out


class FrequencyMasking(BaseAudioAugmentation):
    """Zeros out random frequency channels."""

    def __init__(self, max_mask_freq: int = 16, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.max_mask_freq = max_mask_freq

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 3:  # Spectrogram [B, C, F, T] or [C, F, T]
            num_freq = x.shape[-2]
            f_len = torch.randint(1, min(self.max_mask_freq, num_freq), (1,)).item()
            f_start = torch.randint(0, num_freq - f_len + 1, (1,)).item()
            out = x.clone()
            out[..., f_start : f_start + f_len, :] = 0.0
            return out
        return x


class SpecAugment(BaseAudioAugmentation):
    """Applies combined SpecAugment time and frequency masking."""

    def __init__(self, freq_mask: int = 16, time_mask: int = 4000, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.t_mask = TimeMasking(max_mask_length=time_mask, p=1.0)
        self.f_mask = FrequencyMasking(max_mask_freq=freq_mask, p=1.0)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.t_mask.apply(self.f_mask.apply(x))


class RandomCropping(BaseAudioAugmentation):
    """Crops random contiguous segment of target sample length."""

    def __init__(self, target_samples: int = 64600, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.target_samples = target_samples

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length <= self.target_samples:
            return x
        start = torch.randint(0, length - self.target_samples + 1, (1,)).item()
        return x[..., start : start + self.target_samples]


class RandomShift(BaseAudioAugmentation):
    """Circularly shifts waveform along time dimension."""

    def __init__(self, max_shift: int = 3200, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.max_shift = max_shift

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


class Reverberation(BaseAudioAugmentation):
    """Simulates room reverberation impulse response using exponential decay filter."""

    def __init__(self, decay: float = 0.8, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.decay = decay

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        delay = torch.roll(x, shifts=320, dims=-1) * self.decay
        return x + delay


class CompressionSimulation(BaseAudioAugmentation):
    """Simulates lossy codec compression quantization noise."""

    def __init__(self, bits: int = 8, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.quant = 2 ** (bits - 1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        quantized = torch.round(x * self.quant) / self.quant
        return quantized
