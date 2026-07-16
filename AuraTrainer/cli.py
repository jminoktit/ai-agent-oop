"""AuraTrainer - CLI Entry Point.

This module provides a clean CLI interface for training models.
Usage:
    python -m AuraTrainer.cli --model google/gemma-2-2b-it --epochs 3
"""

import os
import sys
from typing import Optional

from AuraTrainer.core import AuraTrainer, TrainingConfig
from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.CLI")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="AuraTrainer - QLoRA Training")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="Model name")
    parser.add_argument("--dataset-size", type=int, default=100000, help="Dataset size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output", default="./checkpoints", help="Output directory")
    parser.add_argument("--email", default="", help="Notification email")
    parser.add_argument("--hf-token", default="", help="HuggingFace token")

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        dataset_size=args.dataset_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        notify_email=args.email,
        hf_token=args.hf_token or os.getenv("HF_TOKEN", ""),
    )

    trainer = AuraTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
