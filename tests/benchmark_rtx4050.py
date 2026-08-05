"""Benchmark script verifying RTX 4050 throughput optimization and VRAM safety margin."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.models.aasist import AASIST
from app.audio.training.trainer import ProductionAudioTrainer


def benchmark_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    # Micro-batch size 32, raw audio shape (32, 64600)
    batch_size = 32
    num_samples = 320  # 10 batches
    x = torch.randn(num_samples, 64600)
    y = torch.randint(0, 2, (num_samples,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for quick inline synthetic benchmark
        pin_memory=(device.type == "cuda"),
    )

    model = AASIST(num_classes=2)

    config = AudioTrainingConfig(
        epochs=1,
        batch_size=batch_size,
        grad_accum_steps=1,
        use_amp=True,
        checkpoint_dir="./logs/bench_ckpt",
        log_dir="./logs/bench_log",
        tensorboard_dir="./logs/bench_tb",
    )

    trainer = ProductionAudioTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=str(device),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    history = trainer.fit(epochs=1)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    throughput = num_samples / elapsed

    print(f"\n--- Benchmark Results ---")
    print(f"Device           : {device}")
    print(f"Micro Batch Size : {batch_size}")
    print(f"Total Samples    : {num_samples}")
    print(f"Time Elapsed     : {elapsed:.2f} seconds")
    print(f"Throughput       : {throughput:.1f} samples/sec")

    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM        : {peak_vram_gb:.2f} GB")
        assert peak_vram_gb < 5.5, f"VRAM peak {peak_vram_gb:.2f} GB exceeded 5.5 GB safety limit!"
        print("✓ VRAM Safety Check Passed (< 5.5 GB Limit)")

    print("✓ AASIST Optimization Benchmark Successful!")


if __name__ == "__main__":
    benchmark_pipeline()
