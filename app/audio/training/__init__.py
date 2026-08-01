"""Audio training package."""

from app.audio.training.checkpoint import CheckpointManager
from app.audio.training.eer_metrics import compute_biometric_metrics, compute_eer
from app.audio.training.ema import EMAModel
from app.audio.training.loss_factory import ClassBalancedLoss, FocalLoss, LossFactory
from app.audio.training.metrics import AudioMetricsCalculator
from app.audio.training.optimizer import OptimizerFactory
from app.audio.training.optimizers.lion_optimizer import Lion
from app.audio.training.scheduler import SchedulerFactory, WarmupCosineLR
from app.audio.training.trainer import ProductionAudioTrainer, Trainer
from app.audio.training.validator import ValidationEngine

__all__ = [
    "CheckpointManager",
    "compute_eer",
    "compute_biometric_metrics",
    "EMAModel",
    "FocalLoss",
    "ClassBalancedLoss",
    "LossFactory",
    "AudioMetricsCalculator",
    "OptimizerFactory",
    "Lion",
    "SchedulerFactory",
    "WarmupCosineLR",
    "ProductionAudioTrainer",
    "Trainer",
    "ValidationEngine",
]
