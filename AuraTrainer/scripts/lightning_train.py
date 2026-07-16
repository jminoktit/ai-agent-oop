"""Lightning AI Training - AuraTrainer.

This script uses the OOP structure from AuraTrainer.
Deploy: lightning run app lightning_train.py
"""

import os
import sys
import subprocess

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.44.0",
    "trl>=0.9.0",
    "peft>=0.12.0",
    "accelerate>=0.33.0",
    "datasets>=2.20.0",
    "bitsandbytes>=0.43.0",
    "torch>=2.1.0",
    "pyyaml",
    "huggingface_hub",
    "sentencepiece",
    "protobuf",
], check=True)

# Add AuraTrainer to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AuraTrainer.core import AuraTrainer, TrainingConfig


def main():
    """Main entry point for Lightning AI."""
    config = TrainingConfig(
        model_name=os.getenv("MODEL_NAME", "google/gemma-2-2b-it"),
        dataset_size=int(os.getenv("DATASET_SIZE", "100000")),
        num_epochs=int(os.getenv("EPOCHS", "3")),
        learning_rate=float(os.getenv("LR", "2e-4")),
        output_dir=os.getenv("OUTPUT_DIR", "./checkpoints"),
        notify_email=os.getenv("NOTIFY_EMAIL", ""),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_pass=os.getenv("SMTP_PASS", ""),
        hf_token=os.getenv("HF_TOKEN", ""),
    )

    trainer = AuraTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
