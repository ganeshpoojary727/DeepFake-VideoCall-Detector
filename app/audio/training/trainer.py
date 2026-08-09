"""Production Audio Trainer orchestration engine for AASIST model."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.training.checkpoint import CheckpointManager
from app.audio.training.ema import EMAModel
from app.audio.training.loss_factory import LossFactory
from app.audio.training.optimizer import OptimizerFactory
from app.audio.training.scheduler import SchedulerFactory
from app.audio.training.validator import ValidationEngine
from app.audio.utils.audio_logger import get_audio_logger
from app.audio.utils.tensorboard_logger import TensorBoardLogger

logger = get_audio_logger("training.trainer")


def _format_time(seconds: float) -> str:
    """Format duration in seconds into human-readable HH:MM:SS or MM:SS format."""
    if seconds <= 0:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_metrics: Dict[str, Any],
    lr: float,
    epoch_time: float,
    avg_batch_time: float,
    gpu_peak_gb: float,
    checkpoint_status: str,
    best_val_score: float,
) -> None:
    """Print structured end-of-epoch summary card in stdout and log file."""
    val_loss = val_metrics.get("val_loss", 0.0)
    acc = val_metrics.get("accuracy", 0.0)
    if acc <= 1.0 and acc > 0:
        acc *= 100.0
    eer = val_metrics.get("eer", 0.0)
    if eer <= 1.0 and eer > 0:
        eer *= 100.0

    best_score_str = f"{best_val_score:.4f}" if best_val_score != float("inf") else "N/A"

    summary = (
        f"\n"
        f"================================================================================\n"
        f"                           Epoch {epoch}/{total_epochs} Summary                         \n"
        f"================================================================================\n"
        f"  Train Loss            : {train_loss:.4f}\n"
        f"  Validation Loss       : {val_loss:.4f}\n"
        f"  Accuracy              : {acc:.2f}%\n"
        f"  EER                   : {eer:.2f}%\n"
        f"  Learning Rate         : {lr:.2e}\n"
        f"  Epoch Time            : {_format_time(epoch_time)} ({epoch_time:.2f}s)\n"
        f"  Average Batch Time    : {avg_batch_time:.3f}s\n"
        f"  GPU Peak Memory       : {gpu_peak_gb:.2f} GB\n"
        f"  Checkpoint Saved      : {checkpoint_status}\n"
        f"  Best Validation Score : {best_score_str}\n"
        f"================================================================================\n"
    )
    print(summary)
    logger.info(
        "Epoch %d/%d summary — Train Loss: %.4f | Val Loss: %.4f | Acc: %.2f%% | EER: %.2f%% | Best Score: %s",
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        acc,
        eer,
        best_score_str,
    )


class ProductionAudioTrainer:
    """Production training engine for AASIST audio deepfake detector."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[AudioTrainingConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or AudioTrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader or kwargs.get("validation_loader")

        passed_device = kwargs.get("device")
        if passed_device:
            self.device = torch.device(passed_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.device.type == "cuda":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Factories or passed objects
        self.optimizer = kwargs.get("optimizer") or OptimizerFactory(self.config).create_optimizer(self.model)
        self.scheduler = kwargs.get("scheduler") or SchedulerFactory(self.config).create_scheduler(
            self.optimizer, steps_per_epoch=len(train_loader)
        )
        self.criterion = kwargs.get("criterion") or LossFactory(self.config).create_loss()

        # AMP
        passed_use_amp = kwargs.get("use_amp")
        if passed_use_amp is not None:
            self.use_amp = bool(passed_use_amp) and self.device.type == "cuda"
        else:
            self.use_amp = self.config.use_amp and self.device.type == "cuda"

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        # Components
        self.validator = ValidationEngine(self.model, device=str(self.device), use_amp=self.use_amp)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_to_keep=3,
        )
        self.tb_logger = TensorBoardLogger(log_dir=self.config.tensorboard_dir)
        self.ema = EMAModel(self.model, decay=self.config.ema_decay) if self.config.use_ema else None

        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_eer": []}
        self.stats: Dict[str, Any] = {
            "epoch_stats": [],
            "total_training_time": 0.0,
        }
        self.latest_speed_metrics: Dict[str, float] = {}

    def train_epoch(self, epoch: int) -> float:
        """Execute single epoch iteration with gradient accumulation, AMP, real-time tqdm, and live metrics."""
        self.model.train()
        running_loss = 0.0
        successful_steps = 0
        total_steps = len(self.train_loader)

        if total_steps == 0:
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)

        # Speed profiling accumulators
        total_data_time = 0.0
        total_fwd_time = 0.0
        total_bwd_time = 0.0
        total_opt_time = 0.0
        total_batch_time = 0.0

        # GPU metrics caching
        gpu_alloc_gb = 0.0
        gpu_res_gb = 0.0
        gpu_peak_gb = 0.0
        last_gpu_query = 0.0

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            gpu_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            gpu_res_gb = torch.cuda.memory_reserved() / (1024**3)
            gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            last_gpu_query = time.perf_counter()

        pbar = tqdm(
            total=total_steps,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            dynamic_ncols=True,
            unit="batch",
            leave=True,
        )

        train_iter = iter(self.train_loader)
        t_data_start = time.perf_counter()

        try:
            for step in range(total_steps):
                t_batch_start = time.perf_counter()

                # ── First Batch Diagnostic Mode (Epoch 1, Step 0) ──
                if epoch == 1 and step == 0:
                    t0_data = time.perf_counter()
                    batch = next(train_iter)
                    t1_data = time.perf_counter()
                    t_data_load = t1_data - t0_data

                    x_cpu = batch["tensor"] if isinstance(batch, dict) else batch[0]
                    y_cpu = batch["label"] if isinstance(batch, dict) else batch[1]

                    tensor_shape = list(x_cpu.shape)
                    tensor_dtype = str(x_cpu.dtype)
                    tensor_device_before = str(x_cpu.device)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                        gpu_alloc_before = torch.cuda.memory_allocated() / (1024**3)

                    t0_gpu = time.perf_counter()
                    x = x_cpu.to(self.device, non_blocking=True)
                    y = y_cpu.to(self.device, non_blocking=True)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1_gpu = time.perf_counter()
                    t_gpu_transfer = t1_gpu - t0_gpu

                    # Forward pass
                    t0_fwd = time.perf_counter()
                    if self.use_amp and self.scaler is not None:
                        with torch.amp.autocast("cuda"):
                            logits = self.model(x)
                    else:
                        logits = self.model(x)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1_fwd = time.perf_counter()
                    t_fwd = t1_fwd - t0_fwd

                    # Loss calculation
                    t0_loss = time.perf_counter()
                    if self.use_amp and self.scaler is not None:
                        with torch.amp.autocast("cuda"):
                            loss = self.criterion(logits, y) / self.config.grad_accum_steps
                    else:
                        loss = self.criterion(logits, y) / self.config.grad_accum_steps
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1_loss = time.perf_counter()
                    t_loss = t1_loss - t0_loss

                    # Backward pass
                    t0_bwd = time.perf_counter()
                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1_bwd = time.perf_counter()
                    t_bwd = t1_bwd - t0_bwd

                    # Gradient clipping
                    t0_clip = time.perf_counter()
                    grad_norm = 0.0
                    if self.config.gradient_clip_norm > 0:
                        if self.use_amp and self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                        if norm is not None:
                            grad_norm = float(norm)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1_clip = time.perf_counter()
                    t_clip = t1_clip - t0_clip

                    # Optimizer step
                    t0_opt = time.perf_counter()
                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1_opt = time.perf_counter()
                    t_opt = t1_opt - t0_opt

                    if self.ema is not None:
                        self.ema.update(self.model)

                    t_total_first_batch = (
                        t_data_load + t_gpu_transfer + t_fwd + t_loss + t_bwd + t_clip + t_opt
                    )

                    gpu_alloc_after = (
                        torch.cuda.memory_allocated() / (1024**3)
                        if self.device.type == "cuda"
                        else 0.0
                    )
                    gpu_res_after = (
                        torch.cuda.memory_reserved() / (1024**3)
                        if self.device.type == "cuda"
                        else 0.0
                    )
                    gpu_peak_after = (
                        torch.cuda.max_memory_allocated() / (1024**3)
                        if self.device.type == "cuda"
                        else 0.0
                    )

                    diag_output = (
                        f"\nFIRST BATCH DIAGNOSTIC\n"
                        f"----------------------\n"
                        f"Data loading:       {t_data_load:.2f} sec\n"
                        f"Preprocessing:      {t_data_load:.2f} sec (included in data loading)\n"
                        f"Tensor shape:       {tensor_shape}\n"
                        f"Tensor dtype:       {tensor_dtype}\n"
                        f"Tensor device orig: {tensor_device_before}\n"
                        f"GPU transfer:       {t_gpu_transfer:.2f} sec\n"
                        f"AASIST forward:     {t_fwd:.2f} sec\n"
                        f"Loss:               {t_loss:.2f} sec\n"
                        f"Backward:           {t_bwd:.2f} sec\n"
                        f"Gradient clipping:  {t_clip:.2f} sec\n"
                        f"Optimizer step:     {t_opt:.2f} sec\n"
                        f"Total:              {t_total_first_batch:.2f} sec\n\n"
                        f"GPU memory:\n"
                        f"Allocated:          {gpu_alloc_after:.2f} GB\n"
                        f"Reserved:           {gpu_res_after:.2f} GB\n"
                        f"Peak allocated:     {gpu_peak_after:.2f} GB\n"
                    )
                    print(diag_output, flush=True)
                    logger.info(diag_output)

                    running_loss += loss.item() * self.config.grad_accum_steps
                    successful_steps += 1
                    pbar.update(1)
                    t_data_start = time.perf_counter()
                    continue

                t_data = t_batch_start - t_data_start
                total_data_time += t_data

                try:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        break
                    except Exception as fetch_err:
                        logger.error(
                            "Exception during batch loading at step %d in Epoch %d: %s",
                            step + 1,
                            epoch,
                            str(fetch_err),
                            exc_info=True,
                        )
                        pbar.update(1)
                        if not self.config.skip_bad_batches:
                            raise fetch_err
                        t_data_start = time.perf_counter()
                        continue

                    if isinstance(batch, dict):
                        x = batch["tensor"].to(self.device, non_blocking=True)
                        y = batch["label"].to(self.device, non_blocking=True)
                    else:
                        x = batch[0].to(self.device, non_blocking=True)
                        y = batch[1].to(self.device, non_blocking=True)

                    batch_size = x.size(0)
                    grad_norm = 0.0

                    # Forward pass timing
                    t_fwd_start = time.perf_counter()
                    if self.use_amp and self.scaler is not None:
                        with torch.amp.autocast("cuda"):
                            logits = self.model(x)
                    else:
                        logits = self.model(x)

                    # Compute classification loss in FP32 precision outside autocast
                    loss = self.criterion(logits.float(), y.long()) / self.config.grad_accum_steps
                    loss_item = float(loss.item() * self.config.grad_accum_steps)

                    t_fwd = time.perf_counter() - t_fwd_start
                    total_fwd_time += t_fwd

                    if torch.isnan(loss) or torch.isinf(loss) or math.isnan(loss_item) or math.isinf(loss_item):
                        logger.warning("Non-finite loss detected at step %d in Epoch %d (loss=%s). Skipping step.", step + 1, epoch, loss_item)
                        pbar.update(1)
                        t_data_start = time.perf_counter()
                        continue

                    # Backward pass timing
                    t_bwd_start = time.perf_counter()
                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    t_bwd = time.perf_counter() - t_bwd_start
                    total_bwd_time += t_bwd

                    # Optimizer step timing & Gradient clipping
                    t_opt = 0.0
                    if (step + 1) % self.config.grad_accum_steps == 0 or (step + 1) == total_steps:
                        t_opt_start = time.perf_counter()
                        is_log_step = ((step + 1) % self.config.log_interval == 0 or (step + 1) == total_steps)
                        if self.use_amp and self.scaler is not None:
                            if self.config.gradient_clip_norm > 0:
                                self.scaler.unscale_(self.optimizer)
                                norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                                if is_log_step and norm is not None:
                                    grad_norm = float(norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            if self.config.gradient_clip_norm > 0:
                                norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                                if is_log_step and norm is not None:
                                    grad_norm = float(norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        t_opt = time.perf_counter() - t_opt_start
                        total_opt_time += t_opt

                    if self.ema is not None:
                        self.ema.update(self.model)

                    running_loss += loss_item
                    successful_steps += 1

                    t_batch_end = time.perf_counter()
                    batch_time = t_batch_end - t_batch_start
                    total_batch_time += batch_time
                    samples_per_sec = batch_size / batch_time if batch_time > 0 else 0.0

                    # Low overhead GPU memory polling (every gpu_poll_interval_sec, default 3.0s)
                    now = time.perf_counter()
                    if self.device.type == "cuda" and (now - last_gpu_query >= self.config.gpu_poll_interval_sec):
                        gpu_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
                        gpu_res_gb = torch.cuda.memory_reserved() / (1024**3)
                        gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                        last_gpu_query = now

                    curr_lr = self.optimizer.param_groups[0]["lr"]

                    # Explicitly update progress bar for EVERY processed batch
                    pbar.update(1)

                    # Refresh tqdm postfix every 10 batches (or first/last batch of epoch)
                    if (step + 1) % 10 == 0 or (step + 1) == total_steps or step == 0:
                        avg_btime = total_batch_time / max(1, successful_steps)
                        rem_steps = total_steps - (step + 1)
                        eta_epoch = rem_steps * avg_btime
                        rem_epochs = self.config.epochs - epoch
                        total_rem_steps = rem_steps + (rem_epochs * total_steps)
                        eta_train = total_rem_steps * avg_btime
                        accum_step = ((step % self.config.grad_accum_steps) + 1)
                        accum_str = f"{accum_step}/{self.config.grad_accum_steps}"

                        pbar.set_postfix({
                            "Loss": f"{loss_item:.4f}",
                            "LR": f"{curr_lr:.1e}",
                            "GPU": f"{gpu_alloc_gb:.2f}GB",
                            "Batch": f"{batch_time:.2f}s",
                            "Samples/s": f"{samples_per_sec:.1f}",
                            "Accum": accum_str,
                            "ETA_Ep": _format_time(eta_epoch),
                            "ETA_Tr": _format_time(eta_train),
                        })

                    # TensorBoard batch logging every 20 batches (or last batch of epoch)
                    if (step + 1) % 20 == 0 or (step + 1) == total_steps:
                        global_step = (epoch - 1) * total_steps + step
                        self.tb_logger.log_scalar("train_batch/loss", loss_item, global_step)
                        self.tb_logger.log_scalar("train_batch/lr", curr_lr, global_step)
                        self.tb_logger.log_scalar("train_batch/gpu_memory_gb", gpu_alloc_gb, global_step)
                        self.tb_logger.log_scalar("train_batch/samples_per_sec", samples_per_sec, global_step)
                        if grad_norm > 0:
                            self.tb_logger.log_scalar("train_batch/grad_norm", grad_norm, global_step)

                    # Structured log output every log_interval batches (default 50)
                    if (step + 1) % self.config.log_interval == 0 or (step + 1) == total_steps:
                        accum_step = ((step % self.config.grad_accum_steps) + 1)
                        accum_str = f"{accum_step}/{self.config.grad_accum_steps}"
                        logger.info(
                            "Epoch %d [%d/%d] | Loss: %.4f | LR: %.2e | GPU: %.2fGB | Batch: %.2fs | Samples/s: %.1f | Accum: %s",
                            epoch,
                            step + 1,
                            total_steps,
                            loss_item,
                            curr_lr,
                            gpu_alloc_gb,
                            batch_time,
                            samples_per_sec,
                            accum_str,
                        )

                except Exception as e:
                    logger.error(
                        "Exception during execution of Batch %d in Epoch %d: %s",
                        step + 1,
                        epoch,
                        str(e),
                        exc_info=True,
                    )
                    pbar.update(1)
                    if not self.config.skip_bad_batches:
                        raise e

                t_data_start = time.perf_counter()
        finally:
            pbar.close()

        epoch_loss = running_loss / max(1, successful_steps)
        if self.device.type == "cuda":
            gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.empty_cache()

        self.latest_speed_metrics = {
            "data_time": total_data_time,
            "forward_time": total_fwd_time,
            "backward_time": total_bwd_time,
            "optimizer_time": total_opt_time,
            "total_batch_time": total_batch_time,
            "avg_batch_time": total_batch_time / max(1, successful_steps),
            "gpu_peak_gb": gpu_peak_gb,
            "successful_steps": float(successful_steps),
        }
        return float(epoch_loss)

    def train(self) -> Dict[str, Any]:
        """Execute complete multi-epoch training pipeline with comprehensive visibility and monitoring."""
        logger.info("Starting AASIST training — %d epochs, device=%s", self.config.epochs, self.device)

        total_train_start = time.perf_counter()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_epoch(epoch)
            speed_metrics = self.latest_speed_metrics
            epoch_time = time.perf_counter() - epoch_start

            self.history["train_loss"].append(train_loss)

            # Validation pass
            val_metrics = {}
            val_time_sec = 0.0
            if self.val_loader is not None:
                val_metrics = self.validator.evaluate(self.val_loader, self.criterion)
                val_loss = val_metrics.get("val_loss")
                val_acc = val_metrics.get("accuracy")
                val_eer = val_metrics.get("eer")
                val_time_sec = val_metrics.get("val_time_sec", 0.0)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_eer"].append(val_eer)

                saved_best = False
                if val_loss is not None and math.isfinite(val_loss) and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.checkpoint_manager.save_best(self.model, epoch, val_metrics)
                    saved_best = True
                checkpoint_status = "Yes (Best Saved)" if saved_best else "Yes"
            else:
                checkpoint_status = "Yes"

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("val_loss", train_loss))
                    else:
                        self.scheduler.step()

            # Logging & Checkpointing
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb_logger.log_scalar("train/loss", train_loss, epoch)
            self.tb_logger.log_scalar("train/lr", lr, epoch)
            self.tb_logger.log_scalar("train/epoch_time", epoch_time, epoch)
            self.tb_logger.log_scalar("train/gpu_peak_memory_gb", speed_metrics.get("gpu_peak_gb", 0.0), epoch)
            if val_metrics:
                self.tb_logger.log_metrics(val_metrics, epoch, prefix="val")

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metric_value=train_loss,
                history=self.history,
            )

            # Store speed and monitoring statistics
            epoch_stats = {
                "epoch": epoch,
                "epoch_time": epoch_time,
                "val_time": val_time_sec,
                "data_time": speed_metrics.get("data_time", 0.0),
                "forward_time": speed_metrics.get("forward_time", 0.0),
                "backward_time": speed_metrics.get("backward_time", 0.0),
                "optimizer_time": speed_metrics.get("optimizer_time", 0.0),
                "avg_batch_time": speed_metrics.get("avg_batch_time", 0.0),
                "gpu_peak_gb": speed_metrics.get("gpu_peak_gb", 0.0),
            }
            self.stats["epoch_stats"].append(epoch_stats)

            # End-of-Epoch Summary Card
            _print_epoch_summary(
                epoch=epoch,
                total_epochs=self.config.epochs,
                train_loss=train_loss,
                val_metrics=val_metrics,
                lr=lr,
                epoch_time=epoch_time,
                avg_batch_time=speed_metrics.get("avg_batch_time", 0.0),
                gpu_peak_gb=speed_metrics.get("gpu_peak_gb", 0.0),
                checkpoint_status=checkpoint_status,
                best_val_score=self.best_val_loss,
            )

        total_training_time = time.perf_counter() - total_train_start
        self.stats["total_training_time"] = total_training_time
        logger.info("AASIST training completed in %s (%.2fs)", _format_time(total_training_time), total_training_time)

        self.tb_logger.close()
        return self.history

    def fit(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Alias method for training loop execution."""
        if epochs is not None:
            self.config.epochs = epochs
        return self.train()


# Backward compatibility alias
Trainer = ProductionAudioTrainer