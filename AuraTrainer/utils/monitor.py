"""Performance Monitoring Utilities."""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from AuraTrainer.utils.logger import get_logger
from AuraTrainer.utils.gpu import get_vram_usage

logger = get_logger("AuraTrainer.Monitor")


@dataclass
class PerformanceMetrics:
    """Performance metrics for training monitoring."""

    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_percent: float = 0.0
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    gpu_utilization: float = 0.0


class TrainingMonitor:
    """Monitor training performance and resource usage."""

    def __init__(self, log_interval: int = 10, total_steps: Optional[int] = None):
        """Initialize monitor.

        Args:
            log_interval: How often to log metrics (in steps).
            total_steps: Total training steps for ETA calculation.
        """
        self.log_interval = log_interval
        self.total_steps = total_steps
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.samples_seen = 0
        self.tokens_seen = 0
        self.metrics = PerformanceMetrics()

    def update(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        epoch: float,
        batch_size: int = 1,
        num_tokens: int = 0,
    ) -> PerformanceMetrics:
        """Update metrics with new training step data.

        Args:
            step: Current training step.
            loss: Current loss value.
            learning_rate: Current learning rate.
            epoch: Current epoch.
            batch_size: Current batch size.
            num_tokens: Number of tokens in current batch.

        Returns:
            Updated PerformanceMetrics.
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        time_since_last = current_time - self.last_log_time

        self.samples_seen += batch_size
        self.tokens_seen += num_tokens

        self.metrics.step = step
        self.metrics.loss = loss
        self.metrics.learning_rate = learning_rate
        self.metrics.epoch = epoch
        self.metrics.elapsed_seconds = elapsed

        if time_since_last > 0 and step > 0:
            self.metrics.samples_per_sec = batch_size / time_since_last
            self.metrics.tokens_per_sec = num_tokens / time_since_last

        vram = get_vram_usage()
        self.metrics.vram_used_gb = vram["used_mb"] / 1024
        self.metrics.vram_total_gb = vram["total_mb"] / 1024
        if vram["total_mb"] > 0:
            self.metrics.vram_percent = (vram["used_mb"] / vram["total_mb"]) * 100

        if self.total_steps and step > 0:
            steps_remaining = self.total_steps - step
            avg_step_time = elapsed / step
            self.metrics.eta_seconds = steps_remaining * avg_step_time

        try:
            if torch.cuda.is_available():
                self.metrics.gpu_utilization = torch.cuda.utilization()
        except Exception:
            pass

        self.last_log_time = current_time
        return self.metrics

    def should_log(self, step: int) -> bool:
        """Check if metrics should be logged at this step.

        Args:
            step: Current training step.

        Returns:
            True if should log.
        """
        return step % self.log_interval == 0 and step > 0

    def log_metrics(self, metrics: Optional[PerformanceMetrics] = None) -> None:
        """Log current metrics to console.

        Args:
            metrics: Metrics to log (uses last updated if None).
        """
        m = metrics or self.metrics

        eta_str = self._format_time(m.eta_seconds)
        elapsed_str = self._format_time(m.elapsed_seconds)

        logger.info(
            f"Step {m.step:>6d} | "
            f"Loss: {m.loss:.4f} | "
            f"LR: {m.learning_rate:.2e} | "
            f"Epoch: {m.epoch:.2f} | "
            f"Samples/s: {m.samples_per_sec:.1f} | "
            f"Tokens/s: {m.tokens_per_sec:.0f} | "
            f"VRAM: {m.vram_used_gb:.1f}/{m.vram_total_gb:.1f} GB ({m.vram_percent:.0f}%) | "
            f"GPU: {m.gpu_utilization:.0f}% | "
            f"ETA: {eta_str} | "
            f"Elapsed: {elapsed_str}"
        )

    def get_summary(self) -> dict:
        """Get training summary statistics.

        Returns:
            Dictionary with summary metrics.
        """
        return {
            "total_samples": self.samples_seen,
            "total_tokens": self.tokens_seen,
            "total_time_seconds": self.metrics.elapsed_seconds,
            "avg_samples_per_sec": (
                self.samples_seen / self.metrics.elapsed_seconds
                if self.metrics.elapsed_seconds > 0
                else 0
            ),
            "avg_tokens_per_sec": (
                self.tokens_seen / self.metrics.elapsed_seconds
                if self.metrics.elapsed_seconds > 0
                else 0
            ),
            "final_loss": self.metrics.loss,
            "peak_vram_gb": self.metrics.vram_used_gb,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human readable time.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted time string.
        """
        if seconds <= 0:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
