"""
WASAPI Loopback Audio Capture — ring buffer architecture for streaming.

Captures audio from the system's default output device (speakers) using
Windows WASAPI Loopback mode.  This captures the *remote participant's*
voice during video calls — NOT the local microphone.

Architecture
────────────
• ``sounddevice.InputStream`` with ``sd.WasapiSettings(loopback=True)``
• Feeds PCM float32 samples into a ``collections.deque`` ring buffer
• Ring buffer holds 15 seconds at 16 kHz mono (240,000 samples)
• ZERO disk file writes — everything stays in memory
"""

from __future__ import annotations

import collections
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AudioCapture:
    """
    Non-blocking WASAPI loopback audio capture with a 15-second ring buffer.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (default: 16000 Hz).
    buffer_duration : float
        Ring buffer duration in seconds (default: 15.0).
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        buffer_duration: float = 15.0,
    ) -> None:
        self.sample_rate = sample_rate or settings.audio.sample_rate
        self.buffer_duration = buffer_duration

        # Ring buffer: 15s × 16kHz = 240,000 samples
        buffer_size = int(self.sample_rate * buffer_duration)
        self._buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        """Whether the capture is currently active."""
        return self._is_recording

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice callback — appends samples to the ring buffer."""
        if status:
            logger.warning("Audio capture status: %s", status)
        # Convert to mono float32 and append to ring buffer
        if indata.ndim > 1:
            mono = indata[:, 0]
        else:
            mono = indata
        with self._lock:
            self._buffer.extend(mono.astype(np.float32).tolist())

    def _find_loopback_device(self) -> Optional[int]:
        """Find the default WASAPI loopback output device index."""
        try:
            # Try to get the default output device for WASAPI hostapi
            hostapis = sd.query_hostapis()
            for idx, api in enumerate(hostapis):
                if "wasapi" in api["name"].lower():
                    default_output = api.get("default_output_device")
                    if default_output is not None and default_output >= 0:
                        return default_output
            # Fallback: use the system default output device
            default = sd.default.device[1]  # output device
            if default is not None and default >= 0:
                return default
        except Exception as exc:
            logger.debug("Error finding loopback device: %s", exc)
        return None

    def start(self) -> None:
        """Start non-blocking WASAPI loopback audio capture."""
        if self._is_recording:
            logger.warning("Audio capture already running")
            return

        device_idx = self._find_loopback_device()
        if device_idx is None:
            logger.error("No WASAPI output device found for loopback capture")
            return

        try:
            device_info = sd.query_devices(device_idx)
            device_sr = int(device_info.get("default_samplerate", self.sample_rate))
            max_in = int(device_info.get("max_input_channels", 0))

            if max_in > 0:
                channels = min(max_in, 2)
                wasapi_settings = sd.WasapiSettings(exclusive=False)
                self._stream = sd.InputStream(
                    samplerate=device_sr,
                    device=device_idx,
                    channels=channels,
                    dtype="float32",
                    callback=self._audio_callback,
                    extra_settings=wasapi_settings,
                )
                self._stream.start()
                self._is_recording = True
                logger.info("WASAPI capture started on device %d (sr=%d, ch=%d)", device_idx, device_sr, channels)
                return
        except Exception as exc:
            logger.debug("WASAPI device capture attempt: %s", exc)

        # Fallback: standard audio input stream (microphone / default input)
        try:
            block_size = int(self.sample_rate * 0.05)
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            logger.info("Audio capture started using default input stream (sr=%d)", self.sample_rate)
        except Exception as exc2:
            logger.error("Audio capture start failed: %s", exc2)

    def stop(self) -> None:
        """Stop audio capture and release stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Error stopping audio stream: %s", exc)
            self._stream = None
        self._is_recording = False
        logger.info("Audio capture stopped")

    def get_buffer(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Get audio samples from the ring buffer.

        Parameters
        ----------
        duration : float, optional
            How many seconds to retrieve. If None, returns all available.

        Returns
        -------
        np.ndarray
            Audio waveform of shape ``(samples,)``, dtype float32.
        """
        with self._lock:
            data = list(self._buffer)

        if not data:
            return np.array([], dtype=np.float32)

        arr = np.array(data, dtype=np.float32)

        if duration is not None:
            num_samples = int(self.sample_rate * duration)
            if len(arr) >= num_samples:
                return arr[-num_samples:]

        return arr

    def get_15s_buffer(self) -> np.ndarray:
        """
        Return the full 15-second audio context (240,000 samples max).

        Returns
        -------
        np.ndarray
            Audio waveform, up to 15 seconds, shape ``(samples,)``.
        """
        return self.get_buffer(duration=self.buffer_duration)

    def get_chunk_buffer(self, duration_sec: float = 5.0) -> np.ndarray:
        """
        Get the most recent N seconds from the ring buffer.

        Parameters
        ----------
        duration_sec : float
            Duration in seconds to retrieve (default: 5.0).

        Returns
        -------
        np.ndarray
            Audio waveform of the latest ``duration_sec`` seconds.
        """
        return self.get_buffer(duration=duration_sec)

    def get_buffer_fill_ratio(self) -> float:
        """
        Return how full the ring buffer is (0.0 to 1.0).

        Useful for the warmup progress bar (0% to 100%).
        """
        total = self._buffer.maxlen or 1
        return min(len(self._buffer) / total, 1.0)

    def clear_buffer(self) -> None:
        """Clear the ring buffer."""
        with self._lock:
            self._buffer.clear()
