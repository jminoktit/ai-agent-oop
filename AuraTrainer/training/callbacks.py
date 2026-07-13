"""Training Callbacks."""

import time
from typing import Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from AuraTrainer.utils.logger import get_logger
from AuraTrainer.utils.gpu import get_vram_usage
from AuraTrainer.utils.monitor import TrainingMonitor

logger = get_logger("AuraTrainer.Callbacks")


class PerformanceCallback(TrainerCallback):
    """Callback for logging performance metrics."""

    def __init__(self, monitor: Optional[TrainingMonitor] = None):
        """Initialize callback.

        Args:
            monitor: Training monitor instance.
        """
        self.monitor = monitor or TrainingMonitor()
        self.step_start_time = 0.0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at training start."""
        logger.info("Training started - PerformanceCallback active")
        total_steps = state.max_steps if state.max_steps > 0 else None
        self.monitor.total_steps = total_steps

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at step start."""
        self.step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at step end."""
        if self.monitor.should_log(state.global_step):
            logs = state.log_history[-1] if state.log_history else {}
            metrics = self.monitor.update(
                step=state.global_step,
                loss=logs.get("loss", 0.0),
                learning_rate=logs.get("learning_rate", 0.0),
                epoch=logs.get("epoch", 0.0),
            )
            self.monitor.log_metrics(metrics)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at training end."""
        summary = self.monitor.get_summary()
        logger.info("=" * 60)
        logger.info("Training Summary:")
        logger.info(f"  Total samples: {summary['total_samples']:,}")
        logger.info(f"  Total time: {summary['total_time_seconds']:.1f}s")
        logger.info(f"  Avg samples/s: {summary['avg_samples_per_sec']:.2f}")
        logger.info(f"  Final loss: {summary['final_loss']:.4f}")
        logger.info(f"  Peak VRAM: {summary['peak_vram_gb']:.1f} GB")
        logger.info("=" * 60)


class MemoryCallback(TrainerCallback):
    """Callback for monitoring GPU memory and auto-adjusting."""

    def __init__(self, max_vram_percent: float = 95.0):
        """Initialize callback.

        Args:
            max_vram_percent: Maximum VRAM usage before action.
        """
        self.max_vram_percent = max_vram_percent
        self.oom_count = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Check memory usage at each step."""
        vram = get_vram_usage()
        if vram["total_mb"] > 0:
            usage_pct = (vram["used_mb"] / vram["total_mb"]) * 100
            if usage_pct > self.max_vram_percent:
                logger.warning(
                    f"VRAM usage high: {usage_pct:.1f}% "
                    f"({vram['used_mb']:.0f}/{vram['total_mb']:.0f} MB)"
                )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Log memory stats."""
        if logs and "loss" in logs:
            vram = get_vram_usage()
            if vram["total_mb"] > 0:
                logs["vram_used_gb"] = round(vram["used_mb"] / 1024, 2)
                logs["vram_total_gb"] = round(vram["total_mb"] / 1024, 2)


class CheckpointCallback(TrainerCallback):
    """Callback for checkpoint management."""

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        gdrive_dir: Optional[str] = None,
        save_every_n_steps: int = 200,
    ):
        """Initialize callback.

        Args:
            save_dir: Local checkpoint directory.
            gdrive_dir: Google Drive directory for Colab.
            save_every_n_steps: Save frequency.
        """
        self.save_dir = save_dir
        self.gdrive_dir = gdrive_dir
        self.save_every_n_steps = save_every_n_steps

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called when checkpoint is saved."""
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

        if self.gdrive_dir:
            self._copy_to_gdrive(checkpoint_dir)

    def _copy_to_gdrive(self, checkpoint_dir: str) -> None:
        """Copy checkpoint to Google Drive.

        Args:
            checkpoint_dir: Local checkpoint directory.
        """
        try:
            import shutil

            gdrive_path = os.path.join(
                self.gdrive_dir, os.path.basename(checkpoint_dir)
            )
            shutil.copytree(checkpoint_dir, gdrive_path, dirs_exist_ok=True)
            logger.info(f"Checkpoint copied to Google Drive: {gdrive_path}")
        except Exception as e:
            logger.warning(f"Failed to copy checkpoint to Google Drive: {e}")


import os
