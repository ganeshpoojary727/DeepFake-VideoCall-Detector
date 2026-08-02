"""Performance benchmarking module for measuring Latency, FPS, and GPU memory usage."""

from __future__ import annotations

import time
from typing import Dict
import torch


class PerformanceEvaluator:
    """Measures inference runtime metrics: Latency (ms), FPS, and GPU memory usage (MB)."""

    @staticmethod
    def measure_gpu_memory() -> Dict[str, float]:
        """Get GPU memory stats in Megabytes (MB)."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

        dev = torch.cuda.current_device()
        return {
            "allocated_mb": float(torch.cuda.memory_allocated(dev) / (1024 * 1024)),
            "reserved_mb": float(torch.cuda.memory_reserved(dev) / (1024 * 1024)),
            "max_allocated_mb": float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024)),
        }

    @staticmethod
    def benchmark_inference(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark model inference latency and throughput FPS.

        Args:
            model: PyTorch model.
            input_tensor: Input tensor [B, T, C, H, W] or [B, C, H, W].
            num_runs: Benchmark iterations.

        Returns:
            Dict[str, float]: Benchmark dictionary containing latency_ms, fps, and gpu_memory.
        """
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

        avg_latency_sec = total_time / num_runs
        latency_ms = avg_latency_sec * 1000.0
        
        batch_size = input_tensor.size(0)
        num_frames = input_tensor.size(1) if input_tensor.dim() == 5 else 1
        total_frames = batch_size * num_frames * num_runs
        fps = total_frames / total_time if total_time > 0 else 0.0

        gpu_stats = PerformanceEvaluator.measure_gpu_memory()

        return {
            "latency_ms": float(latency_ms),
            "fps": float(fps),
            "gpu_memory_allocated_mb": gpu_stats["allocated_mb"],
            "gpu_memory_reserved_mb": gpu_stats["reserved_mb"],
        }
