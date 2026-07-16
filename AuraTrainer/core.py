"""AuraTrainer - Main OOP Module.

This module provides the main AuraTrainer class that orchestrates
the entire training pipeline using OOP principles.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from AuraTrainer.utils.logger import get_logger
from AuraTrainer.utils.gpu import GPUManager, GPUConfig, auto_configure
from AuraTrainer.models.loader import ModelLoader
from AuraTrainer.models.lora import LoRAConfigurator
from AuraTrainer.data.loader import DatasetLoader
from AuraTrainer.data.formatter import DatasetFormatter
from AuraTrainer.data.sampler import DatasetSampler
from AuraTrainer.training.trainer import AuraTrainer as Trainer
from AuraTrainer.training.callbacks import PerformanceCallback, MemoryCallback

logger = get_logger("AuraTrainer.Core")


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    model_name: str = "google/gemma-2-2b-it"
    dataset_size: int = 100000
    batch_size: int = 10000
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    output_dir: str = "./checkpoints"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    notify_email: str = ""
    smtp_user: str = ""
    smtp_pass: str = ""
    hf_token: str = ""


class AuraTrainer:
    """Main AuraTrainer class that orchestrates the entire training pipeline."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize AuraTrainer.

        Args:
            config: Training configuration. If None, uses defaults.
        """
        self.config = config or TrainingConfig()
        self.gpu_config: Optional[GPUConfig] = None
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Initialize components
        self.gpu_manager = GPUManager()
        self.model_loader = None
        self.lora_configurator = None
        self.dataset_loader = DatasetLoader()
        self.data_formatter = DatasetFormatter()
        self.data_sampler = DatasetSampler()

    def detect_gpu(self) -> GPUConfig:
        """Detect and configure GPU.

        Returns:
            GPU configuration.
        """
        self.gpu_config = auto_configure()
        return self.gpu_config

    def load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.model_loader = ModelLoader(
            model_name=self.config.model_name,
            max_length=self.config.max_seq_length,
        )

        self.model, self.tokenizer = self.model_loader.load_for_training(
            quantize=True,
            bits=4,
            compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        logger.info("Model loaded successfully")

    def apply_lora(self) -> None:
        """Apply LoRA to model."""
        logger.info("Applying LoRA...")

        self.lora_configurator = LoRAConfigurator(
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )

        self.model = self.lora_configurator.apply_to_model(self.model)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def load_dataset(self) -> Any:
        """Load and format dataset.

        Returns:
            Formatted dataset.
        """
        logger.info(f"Loading dataset (size: {self.config.dataset_size})...")

        # Load datasets
        arabic_data = self.dataset_loader.load_single(
            name="arabic",
            repo="arbml/alpaca_arabic",
            split="train",
            streaming=False,
        )

        python_data = self.dataset_loader.load_single(
            name="python",
            repo="code-search-net/code_search_net",
            subset="python",
            split="train",
            streaming=False,
        )

        # Sample datasets
        arabic_sample = self.data_sampler.sample_dataset(
            list(arabic_data),
            num_samples=min(self.config.dataset_size // 2, len(arabic_data)),
            category="arabic",
        )

        python_sample = self.data_sampler.sample_dataset(
            list(python_data),
            num_samples=min(self.config.dataset_size // 2, len(python_data)),
            category="python",
        )

        # Format datasets
        arabic_formatted = [self.data_formatter.format_arabic(ex) for ex in arabic_sample]
        python_formatted = [self.data_formatter.format_chat(ex) for ex in python_sample]

        # Merge datasets
        from datasets import Dataset
        all_data = [{"text": text} for text in arabic_formatted + python_formatted]
        dataset = Dataset.from_list(all_data).shuffle(seed=42)

        logger.info(f"Dataset loaded: {len(dataset)} samples")
        return dataset

    def train(self, dataset: Any) -> Dict[str, Any]:
        """Train the model.

        Args:
            dataset: Training dataset.

        Returns:
            Training metrics.
        """
        logger.info("Starting training...")

        training_config = {
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 3,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp16": not torch.cuda.is_bf16_supported(),
            "optim": "paged_adamw_32bit",
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 0.3,
            "report_to": "none",
            "output_dir": self.config.output_dir,
            "max_length": self.config.max_seq_length,
        }

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            gpu_config=self.gpu_config,
            training_config=training_config,
        )

        metrics = self.trainer.train()
        logger.info(f"Training completed! Loss: {metrics.get('train_loss', 'N/A')}")
        return metrics

    def save_model(self) -> None:
        """Save the trained model."""
        if self.trainer:
            self.trainer.save_model(self.config.output_dir)
            logger.info(f"Model saved to {self.config.output_dir}")

    def send_notification(self, metrics: Dict[str, Any]) -> None:
        """Send email notification.

        Args:
            metrics: Training metrics.
        """
        if not self.config.notify_email:
            return

        import smtplib
        from email.mime.text import MIMEText

        if not self.config.smtp_user or not self.config.smtp_pass:
            logger.warning("SMTP not configured, skipping email")
            return

        subject = f"✅ Training Completed - {self.config.model_name}"
        body = (
            f"Training completed successfully!\n\n"
            f"Model: {self.config.model_name}\n"
            f"Loss: {metrics.get('train_loss', 'N/A')}\n"
            f"Runtime: {metrics.get('train_runtime', 0):.1f}s\n"
        )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.config.smtp_user
        msg["To"] = self.config.notify_email

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls()
                s.login(self.config.smtp_user, self.config.smtp_pass)
                s.send_message(msg)
            logger.info(f"Email sent to {self.config.notify_email}")
        except Exception as e:
            logger.error(f"Email failed: {e}")

    def run(self) -> Dict[str, Any]:
        """Run the full training pipeline.

        Returns:
            Training metrics.
        """
        logger.info("=" * 60)
        logger.info("AuraTrainer - Starting Training Pipeline")
        logger.info("=" * 60)

        # Detect GPU
        self.detect_gpu()

        # Load model
        self.load_model()

        # Apply LoRA
        self.apply_lora()

        # Load dataset
        dataset = self.load_dataset()

        # Train
        metrics = self.train(dataset)

        # Save model
        self.save_model()

        # Send notification
        self.send_notification(metrics)

        logger.info("=" * 60)
        logger.info("Training pipeline completed!")
        logger.info("=" * 60)

        return metrics
