"""Checkpoint Management."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.Checkpoint")


class CheckpointManager:
    """Manage training checkpoints for resumption."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_optimizer: bool = True,
        max_checkpoints: int = 3,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints.
            save_optimizer: Whether to save optimizer state.
            max_checkpoints: Maximum checkpoints to keep.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_optimizer = save_optimizer
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the directory.

        Returns:
            Path to latest checkpoint, or None if no checkpoint found.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoint found, starting from scratch")
            return None

        latest = checkpoints[-1]
        logger.info(f"Found checkpoint: {latest}")
        return str(latest)

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoints sorted by step number.

        Returns:
            List of checkpoint paths sorted chronologically.
        """
        checkpoints = []
        if not self.checkpoint_dir.exists():
            return checkpoints

        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (ValueError, IndexError):
                    continue

        checkpoints.sort(key=lambda x: x[0])
        return [c[1] for c in checkpoints]

    def save_checkpoint(
        self,
        model,
        tokenizer,
        optimizer=None,
        scheduler=None,
        step: int = 0,
        metrics: Optional[Dict] = None,
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: Model to save.
            tokenizer: Tokenizer to save.
            optimizer: Optional optimizer to save.
            scheduler: Optional scheduler to save.
            step: Current training step.
            metrics: Optional metrics to save.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        if optimizer and self.save_optimizer:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)

        if scheduler:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)

        state = {
            "step": step,
            "metrics": metrics or {},
        }
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint_state(self, checkpoint_path: str) -> Dict:
        """Load checkpoint metadata.

        Args:
            checkpoint_path: Path to checkpoint.

        Returns:
            Checkpoint state dictionary.
        """
        state_path = Path(checkpoint_path) / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                return json.load(f)
        return {"step": 0, "metrics": {}}

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max limit."""
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {oldest}")
            shutil.rmtree(oldest)

    def export_merged_model(
        self,
        peft_model,
        output_dir: str,
        safe_serialization: bool = True,
    ) -> str:
        """Export merged model (LoRA + base).

        Args:
            peft_model: Model with LoRA adapters.
            output_dir: Output directory.
            safe_serialization: Use safetensors.

        Returns:
            Path to merged model.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(
            str(output_path), safe_serialization=safe_serialization
        )

        logger.info("Merged model exported successfully")
        return str(output_path)

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """Get information about a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.

        Returns:
            Dictionary with checkpoint info.
        """
        path = Path(checkpoint_path)
        info = {
            "path": str(path),
            "exists": path.exists(),
            "files": [],
            "total_size_mb": 0,
        }

        if path.exists():
            for file in path.rglob("*"):
                if file.is_file():
                    size = file.stat().st_size
                    info["files"].append({
                        "name": file.name,
                        "size_mb": round(size / (1024 * 1024), 2),
                    })
                    info["total_size_mb"] += size

            info["total_size_mb"] = round(info["total_size_mb"] / (1024 * 1024), 2)

        return info
