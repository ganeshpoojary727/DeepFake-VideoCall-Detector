"""Production Video Trainer orchestration engine mirroring Audio subsystem."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.video.configs.training_config import VideoTrainingConfig
from app.video.core.base_trainer import BaseTrainer
from app.video.datasets.dataloader import video_collate_fn
from app.video.evaluation.metrics import EvaluationMetrics
from app.video.exceptions.video_exceptions import TrainingError
from app.video.training.checkpoint_manager import CheckpointManager
from app.video.training.early_stopping import EarlyStopping
from app.video.training.loss_factory import LossFactory
from app.video.training.metrics import VideoMetricsCalculator
from app.video.training.mixed_precision import MixedPrecisionHandler
from app.video.training.optimizer_factory import OptimizerFactory
from app.video.training.scheduler_factory import SchedulerFactory

logger = logging.getLogger(__name__)


def _print_best_model_updated(criterion: str, previous_val: str, new_val: str, checkpoint_path: str) -> None:
    """Print structured notification when a model checkpoint improves on a evaluation criterion."""
    msg = (
        f"\nBEST MODEL UPDATED\n"
        f"Criterion: {criterion}\n"
        f"Previous: {previous_val}\n"
        f"New:      {new_val}\n"
        f"Checkpoint: {checkpoint_path}\n"
    )
    print(msg, flush=True)
    logger.info(msg.strip())


def _print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    epoch_time: float,
    gpu_peak_gb: float,
) -> None:
    """Print structured end-of-epoch summary card."""
    t_loss = train_metrics.get("loss", 0.0)
    t_acc = train_metrics.get("accuracy", 0.0)
    t_prec = train_metrics.get("precision", 0.0)
    t_rec = train_metrics.get("recall", 0.0)
    t_f1 = train_metrics.get("f1", 0.0)

    v_loss = val_metrics.get("val_loss", 0.0)
    v_acc = val_metrics.get("accuracy", 0.0)
    v_prec = val_metrics.get("precision", 0.0)
    v_rec = val_metrics.get("recall", 0.0)
    v_f1 = val_metrics.get("f1", 0.0)
    v_auc = val_metrics.get("auc", 0.0)

    summary = (
        f"\n============================================================\n"
        f"Epoch {epoch:02d}/{total_epochs:02d}\n"
        f"Train Loss:      {t_loss:.4f}\n"
        f"Train Accuracy:  {t_acc:.4f}\n"
        f"Train Precision: {t_prec:.4f}\n"
        f"Train Recall:    {t_rec:.4f}\n"
        f"Train F1:        {t_f1:.4f}\n\n"
        f"Val Loss:        {v_loss:.4f}\n"
        f"Val Accuracy:    {v_acc:.4f}\n"
        f"Val Precision:   {v_prec:.4f}\n"
        f"Val Recall:      {v_rec:.4f}\n"
        f"Val F1:          {v_f1:.4f}\n"
        f"Val AUC:         {v_auc:.4f}\n\n"
        f"Learning Rate:   {lr:.5e}\n"
        f"Epoch Time:      {epoch_time:.1f}s\n"
        f"GPU Memory:      {gpu_peak_gb:.2f} GB\n"
        f"============================================================\n"
    )
    print(summary, flush=True)
    logger.info(summary.strip())


class ProductionVideoTrainer(BaseTrainer):
    """Production training engine for video deepfake detector models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[VideoTrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config or VideoTrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(self.config.device if torch.cuda.is_available() and "cuda" in self.config.device else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer or OptimizerFactory.create_optimizer(self.model, config=self.config)
        self.loss_fn = loss_fn or LossFactory.create_loss(config=self.config)
        self.scheduler = scheduler or SchedulerFactory.create_scheduler(self.optimizer, config=self.config)

        self.amp_handler = MixedPrecisionHandler(enabled=self.config.use_amp, device=str(self.device))
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            save_top_k=self.config.save_top_k,
        )

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_auc = 0.0

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auc": [],
            "epoch_duration_sec": [],
            "gpu_vram_mb": [],
        }
        self.latest_train_metrics: Dict[str, float] = {}

    def train_epoch(self, dataloader_or_epoch: Optional[Union[DataLoader, int]] = None) -> float:
        """Execute single training epoch iteration with AMP, gradient accumulation, and live tqdm progress."""
        loader = self.train_loader
        epoch_idx = 1
        if isinstance(dataloader_or_epoch, DataLoader):
            loader = dataloader_or_epoch
        elif isinstance(dataloader_or_epoch, int):
            epoch_idx = dataloader_or_epoch

        if loader is None:
            raise TrainingError("train_loader is not set for trainer.")

        total_steps = len(loader)
        if total_steps == 0:
            return 0.0

        self.model.train()
        running_loss = 0.0
        total_samples = 0
        correct_samples = 0
        accum_steps = self.config.gradient_accumulation_steps

        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        self.optimizer.zero_grad(set_to_none=True)

        gpu_alloc_gb = 0.0
        gpu_res_gb = 0.0
        gpu_peak_gb = 0.0
        last_gpu_query = 0.0
        last_pbar_time = 0.0

        if self.device.type == "cuda":
            gpu_alloc_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
            gpu_res_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
            gpu_peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            last_gpu_query = time.perf_counter()

        pbar = tqdm(
            total=total_steps,
            desc=f"Epoch {epoch_idx:02d}/{self.config.epochs:02d} [Train]",
            dynamic_ncols=True,
            unit="batch",
            leave=True,
            mininterval=0.5,
        )

        for step, batch in enumerate(loader):
            t_batch_start = time.perf_counter()
            if isinstance(batch, dict):
                x = batch["tensor"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
            elif hasattr(batch, "tensor"):
                x = batch.tensor.to(self.device)
                y = torch.tensor([batch.label], device=self.device)
            elif isinstance(batch, (tuple, list)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            else:
                pbar.update(1)
                continue

            with self.amp_handler.autocast():
                logits = self.model(x)
                loss = self.loss_fn(logits, y) / accum_steps

            self.amp_handler.scale_and_step(
                loss=loss,
                optimizer=self.optimizer,
                clip_norm=self.config.gradient_clip_norm if (step + 1) % accum_steps == 0 else 0.0,
                model=self.model,
            )

            if (step + 1) % accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = x.size(0)
            loss_val = float(loss.item() * accum_steps)
            running_loss += loss_val * batch_size
            total_samples += batch_size

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)[:, 1] if logits.size(-1) > 1 else torch.sigmoid(logits)
                preds = torch.argmax(logits, dim=-1) if logits.size(-1) > 1 else (logits > 0.0).long()
                correct_samples += (preds == y).sum().item()

                all_probs.append(probs.detach().cpu())
                all_labels.append(y.detach().cpu())

            t_batch_end = time.perf_counter()
            batch_dur = t_batch_end - t_batch_start

            # Query VRAM periodically (every 3.0 seconds) without forced GPU synchronization
            now = time.perf_counter()
            if self.device.type == "cuda" and (now - last_gpu_query >= 3.0):
                gpu_alloc_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
                gpu_res_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
                gpu_peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                last_gpu_query = now

            curr_lr = self.optimizer.param_groups[0]["lr"]
            running_loss_avg = (running_loss / total_samples) if total_samples > 0 else 0.0
            running_acc = (correct_samples / total_samples) if total_samples > 0 else 0.0

            pbar.update(1)

            # Smooth rate-limited update (every 5 batches, >= 1.0s, or last step) to keep terminal responsive
            if (step + 1) % 5 == 0 or (step + 1) == total_steps or (now - last_pbar_time >= 1.0):
                last_pbar_time = now
                pbar.set_postfix({
                    "loss": f"{running_loss_avg:.4f}",
                    "acc": f"{running_acc:.4f}",
                    "VRAM": f"{gpu_res_gb:.1f}GB",
                    "step": f"{batch_dur:.1f}s",
                })

            # Structured logger info every 50 batches
            if (step + 1) % 50 == 0 or (step + 1) == total_steps:
                logger.info(
                    "Epoch %02d/%02d [%d/%d] | Loss: %.4f | Acc: %.4f | LR: %.2e | VRAM: %.2fGB (Peak: %.2fGB, Alloc: %.2fGB) | %.1fs/step",
                    epoch_idx,
                    self.config.epochs,
                    step + 1,
                    total_steps,
                    running_loss_avg,
                    running_acc,
                    curr_lr,
                    gpu_res_gb,
                    gpu_peak_gb,
                    gpu_alloc_gb,
                    batch_dur,
                )

        pbar.close()

        epoch_loss = (running_loss / total_samples) if total_samples > 0 else 0.0

        if all_probs:
            y_true = torch.cat(all_labels, dim=0).numpy()
            y_probs = torch.cat(all_probs, dim=0).numpy()
            train_eval = EvaluationMetrics.compute_all(y_true, y_probs)
        else:
            train_eval = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        train_eval["loss"] = float(epoch_loss)
        self.latest_train_metrics = train_eval

        return float(epoch_loss)

    def validate(
        self,
        dataloader_or_dataset: Optional[Union[DataLoader, Dataset]] = None,
        batch_size: int = 4,
        epoch: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Execute evaluation pass over validation dataset split with visible tqdm progress."""
        loader = self.val_loader

        if isinstance(dataloader_or_dataset, DataLoader):
            loader = dataloader_or_dataset
        elif isinstance(dataloader_or_dataset, Dataset):
            loader = DataLoader(
                dataloader_or_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=video_collate_fn,
            )

        if loader is None:
            return {"val_loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []
        total_val_samples = 0
        correct_val_samples = 0

        total_batches = len(loader)
        if max_batches is not None and max_batches > 0:
            total_batches = min(total_batches, max_batches)

        ep_desc = f"Epoch {epoch:02d} [Val]" if epoch is not None else "Validation"
        pbar = tqdm(
            total=total_batches,
            desc=ep_desc,
            dynamic_ncols=True,
            unit="batch",
            leave=False,
            mininterval=0.5,
        )

        with torch.inference_mode():
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break

                if isinstance(batch, dict):
                    x = batch["tensor"].to(self.device, non_blocking=True)
                    y = batch["label"].to(self.device, non_blocking=True)
                elif hasattr(batch, "tensor"):
                    x = batch.tensor.to(self.device)
                    y = torch.tensor([batch.label], device=self.device)
                elif isinstance(batch, (tuple, list)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    pbar.update(1)
                    continue

                with self.amp_handler.autocast():
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                b_size = x.size(0)
                running_loss += loss.item() * b_size
                total_val_samples += b_size

                probs = torch.softmax(logits, dim=-1)[:, 1] if logits.size(-1) > 1 else torch.sigmoid(logits)
                preds = torch.argmax(logits, dim=-1) if logits.size(-1) > 1 else (logits > 0.0).long()
                correct_val_samples += (preds == y).sum().item()

                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

                pbar.update(1)
                cur_vloss = running_loss / max(1, total_val_samples)
                cur_vacc = correct_val_samples / max(1, total_val_samples)
                pbar.set_postfix({"val_loss": f"{cur_vloss:.4f}", "val_acc": f"{cur_vacc:.4f}"})

        pbar.close()

        val_loss = (running_loss / total_val_samples) if total_val_samples > 0 else 0.0

        if all_probs:
            y_true = torch.cat(all_labels, dim=0).numpy()
            y_probs = torch.cat(all_probs, dim=0).numpy()
            eval_metrics = EvaluationMetrics.compute_all(y_true, y_probs)
        else:
            eval_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        eval_metrics["val_loss"] = float(val_loss)
        return eval_metrics

    def save_checkpoint(self, epoch: int, path: Optional[Union[Path, str]] = None) -> Path:
        """Persist trainer and model state checkpoint to disk."""
        filename = Path(path).name if path is not None else f"checkpoint_epoch_{epoch:03d}.pt"
        return self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=self.best_val_loss,
            metrics=self.history,
            scheduler=self.scheduler,
            filename=filename,
        )

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Resume training state from saved checkpoint file."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        chk = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(chk, dict):
            if "model" in chk:
                self.model.load_state_dict(chk["model"])
            elif "state_dict" in chk:
                self.model.load_state_dict(chk["state_dict"])

            if "optimizer" in chk and self.optimizer is not None:
                self.optimizer.load_state_dict(chk["optimizer"])

            if "scheduler" in chk and self.scheduler is not None and chk["scheduler"] is not None:
                try:
                    self.scheduler.load_state_dict(chk["scheduler"])
                except Exception as e:
                    logger.warning(f"Could not restore scheduler state: {e}")

            if "metrics" in chk and isinstance(chk["metrics"], dict):
                self.history = chk["metrics"]
                if self.history.get("val_loss"):
                    self.best_val_loss = min(self.history["val_loss"])
                if self.history.get("val_accuracy"):
                    self.best_accuracy = max(self.history["val_accuracy"])
                if self.history.get("val_f1"):
                    self.best_f1 = max(self.history["val_f1"])
                if self.history.get("val_auc"):
                    self.best_auc = max(self.history["val_auc"])

            self.start_epoch = chk.get("epoch", 0)
            return self.start_epoch
        return 0

    def train(self) -> Dict[str, Any]:
        """Execute complete multi-epoch training pipeline with progress bar and summary cards."""
        start_ep = getattr(self, "start_epoch", 0)
        logger.info(f"Executing training loop for epochs {start_ep + 1} to {self.config.epochs}...")

        for epoch in range(start_ep + 1, self.config.epochs + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            t0 = time.perf_counter()
            t_loss = self.train_epoch(epoch)

            t_metrics = self.latest_train_metrics
            self.history["train_loss"].append(t_metrics.get("loss", t_loss))
            self.history["train_accuracy"].append(t_metrics.get("accuracy", 0.0))
            self.history["train_precision"].append(t_metrics.get("precision", 0.0))
            self.history["train_recall"].append(t_metrics.get("recall", 0.0))
            self.history["train_f1"].append(t_metrics.get("f1", 0.0))

            val_metrics: Dict[str, float] = {}
            if self.val_loader is not None:
                val_max_batches = getattr(self.config, "val_max_batches", None)
                is_final_epoch = (epoch == self.config.epochs)
                max_batches = None if is_final_epoch else val_max_batches
                val_metrics = self.validate(epoch=epoch, max_batches=max_batches)
                v_loss = val_metrics.get("val_loss", 0.0)
                v_acc = val_metrics.get("accuracy", 0.0)
                v_prec = val_metrics.get("precision", 0.0)
                v_rec = val_metrics.get("recall", 0.0)
                v_f1 = val_metrics.get("f1", 0.0)
                v_auc = val_metrics.get("auc", 0.0)

                self.history["val_loss"].append(v_loss)
                self.history["val_accuracy"].append(v_acc)
                self.history["val_precision"].append(v_prec)
                self.history["val_recall"].append(v_rec)
                self.history["val_f1"].append(v_f1)
                self.history["val_auc"].append(v_auc)

            t1 = time.perf_counter()
            dur = t1 - t0
            vram_gb = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                if torch.cuda.is_available() and "cuda" in str(self.device)
                else 0.0
            )

            self.history["epoch_duration_sec"].append(round(dur, 2))
            self.history["gpu_vram_mb"].append(round(vram_gb * 1024, 2))

            curr_lr = self.optimizer.param_groups[0]["lr"]

            # Print structured end-of-epoch summary card
            _print_epoch_summary(
                epoch=epoch,
                total_epochs=self.config.epochs,
                train_metrics=t_metrics,
                val_metrics=val_metrics,
                lr=curr_lr,
                epoch_time=dur,
                gpu_peak_gb=vram_gb,
            )

            # Save latest checkpoint and incrementally persist training_history.json
            self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "latest.pt")
            try:
                import json
                history_path = self.config.checkpoint_dir / "training_history.json"
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(self.history, f, indent=2)
            except Exception as hist_err:
                logger.warning(f"Could not persist training_history.json: {hist_err}")

            if self.val_loader is not None:
                v_loss = val_metrics.get("val_loss", 0.0)
                v_acc = val_metrics.get("accuracy", 0.0)
                v_f1 = val_metrics.get("f1", 0.0)
                v_auc = val_metrics.get("auc", 0.0)

                if v_loss < self.best_val_loss:
                    prev = self.best_val_loss
                    self.best_val_loss = v_loss
                    ckpt_p = self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_loss.pt")
                    self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_model.pt")
                    prev_str = f"{prev:.4f}" if prev != float("inf") else "inf"
                    _print_best_model_updated("validation loss", prev_str, f"{v_loss:.4f}", str(ckpt_p))

                if v_acc > self.best_accuracy:
                    prev = self.best_accuracy
                    self.best_accuracy = v_acc
                    ckpt_p = self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_accuracy.pt")
                    _print_best_model_updated("validation accuracy", f"{prev:.4f}", f"{v_acc:.4f}", str(ckpt_p))

                if v_f1 > self.best_f1:
                    prev = self.best_f1
                    self.best_f1 = v_f1
                    ckpt_p = self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_f1.pt")
                    _print_best_model_updated("validation F1", f"{prev:.4f}", f"{v_f1:.4f}", str(ckpt_p))

                if v_auc > self.best_auc:
                    prev = self.best_auc
                    self.best_auc = v_auc
                    ckpt_p = self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_auc.pt")
                    _print_best_model_updated("validation AUC", f"{prev:.4f}", f"{v_auc:.4f}", str(ckpt_p))

                if self.early_stopping(v_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("val_loss", t_loss) if self.val_loader else t_loss)
                    else:
                        self.scheduler.step()

        return self.history


# Class aliases
VideoTrainer = ProductionVideoTrainer
Trainer = ProductionVideoTrainer
