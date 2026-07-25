"""
Feature Extraction Module

Responsible for:
- Creating Mel Spectrograms
- Converting to Decibel Scale
- Resizing spectrograms
- Converting to PyTorch tensors
"""

import numpy as np
import librosa
import torch

from app.config.settings import settings

class FeatureExtractor:
    """
    Extracts CNN-ready features from
    preprocessed audio.
    """

    def __init__(self):

        self.sample_rate = settings.SAMPLE_RATE

        self.n_fft = 2048

        self.hop_length = 512

        self.n_mels = 128

        self.target_length = 100

    def create_mel_spectrogram(
          self,
          audio: np.ndarray
          ):
      """
      Generate Mel Spectrogram.
      """
      mel = librosa.feature.melspectrogram(
        y=audio,
        sr=self.sample_rate,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        n_mels=self.n_mels
      )
      return mel
    def convert_to_db(
          self,
        mel_spectrogram: np.ndarray
        ):
      """
      Convert Mel Spectrogram
      to Decibel Scale.
      """

      mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )
      return mel_db
    def resize_spectrogram(
      self,
      mel_db: np.ndarray
      ):
      """
       Resize Mel Spectrogram
       to fixed size.
      """

      if mel_db.shape[1] < self.target_length:

        pad_width = (
            self.target_length
            - mel_db.shape[1]
        )

        mel_db = np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

      else:

        mel_db = mel_db[:, :self.target_length]

      return mel_db

    def to_tensor(
      self,
      mel_db: np.ndarray
       ):
     """
      Convert spectrogram
      to PyTorch Tensor.
     """

     mel_tensor = torch.tensor(
      mel_db,
      dtype=torch.float32,
      device=settings.DEVICE
)

     mel_tensor = mel_tensor.unsqueeze(0)
     return mel_tensor
    def extract(
      self,
      audio: np.ndarray
      ):
      """
     Complete Feature Extraction Pipeline.
     """

      mel = self.create_mel_spectrogram(audio)
 
      mel_db = self.convert_to_db(mel)

      mel_db = self.resize_spectrogram(mel_db)

      mel_tensor = self.to_tensor(mel_db)

      return mel_tensor