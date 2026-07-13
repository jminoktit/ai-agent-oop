"""QLoRA Training Pipeline."""

import os
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from AuraTrainer.utils.logger import get_logger
from AuraTrainer.utils.monitor import TrainingMonitor
from AuraTrainer.utils.gpu import GPUConfig

logger = get_logger("AuraTrainer.Trainer")


class AuraTrainer:
    """Main training orchestrator for AuraBook QLoRA fine-tuning."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        gpu_config: Optional[GPUConfig] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            tokenizer: Tokenizer.
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            gpu_config: GPU configuration.
            training_config: Training hyperparameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.gpu_config = gpu_config
        self.config = training_config or {}
        self.monitor = TrainingMonitor(
            log_interval=self.config.get("logging_steps", 10)
        )
        self.trainer = None

    def _create_training_args(self) -> SFTConfig:
        """Create training arguments.

        Returns:
            SFTConfig instance.
        """
        output_dir = self.config.get("output_dir", "./outputs")

        args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_train_epochs", 3),
            max_steps=self.config.get("max_steps", -1),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            warmup_steps=self.config.get("warmup_steps", 100),
            weight_decay=self.config.get("weight_decay", 0.01),
            adam_beta1=self.config.get("adam_beta1", 0.9),
            adam_beta2=self.config.get("adam_beta2", 0.999),
            adam_epsilon=self.config.get("adam_epsilon", 1e-8),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            fp16=self.config.get("fp16", False),
            bf16=self.config.get("bf16", True),
            tf32=self.config.get("tf32", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            optim=self.config.get("optim", "paged_adamw_8bit"),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 200),
            save_total_limit=self.config.get("save_total_limit", 3),
            eval_strategy=self.config.get("eval_strategy", "steps"),
            eval_steps=self.config.get("eval_steps", 200),
            dataloader_num_workers=self.config.get("dataloader_num_workers", 4),
            remove_unused_columns=self.config.get("remove_unused_columns", False),
            report_to=self.config.get("report_to", "none"),
            run_name=self.config.get("run_name", "aura-book-qlora"),
            seed=self.config.get("seed", 42),
            ddp_timeout=self.config.get("ddp_timeout", 1800),
            dataset_text_field=None,
            max_seq_length=self.config.get("max_length", 2048),
            packing=False,
        )

        if self.gpu_config:
            if self.gpu_config.platform and "colab" in str(self.gpu_config.platform):
                gdrive_enabled = self.config.get("gdrive", {}).get("enabled", False)
                if gdrive_enabled:
                    gdrive_dir = self.config.get("gdrive", {}).get("save_dir", "")
                    if gdrive_dir:
                        args.output_dir = gdrive_dir

        return args

    def _create_data_collator(self) -> DataCollatorForLanguageModeling:
        """Create data collator.

        Returns:
            DataCollatorForLanguageModeling instance.
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """Start or resume training.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            Training metrics dictionary.
        """
        training_args = self._create_training_args()
        data_collator = self._create_data_collator()

        total_steps = (
            len(self.train_dataset)
            // training_args.per_device_train_batch_size
            // training_args.gradient_accumulation_steps
            * int(training_args.num_train_epochs)
        )
        self.monitor.total_steps = total_steps

        logger.info("=" * 60)
        logger.info("Starting AuraBook QLoRA Training")
        logger.info("=" * 60)
        logger.info(f"Dataset size: {len(self.train_dataset):,} samples")
        logger.info(f"Total steps: {total_steps:,}")
        logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"Grad accum: {training_args.gradient_accumulation_steps}")
        logger.info(f"Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        logger.info(f"Epochs: {training_args.num_train_epochs}")
        logger.info(f"BF16: {training_args.bf16}, FP16: {training_args.fp16}")

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )

        try:
            if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
                result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                result = self.trainer.train()

            metrics = result.metrics
            logger.info("=" * 60)
            logger.info("Training completed successfully!")
            logger.info(f"Train loss: {metrics.get('train_loss', 'N/A')}")
            logger.info(f"Train runtime: {metrics.get('train_runtime', 0):.1f}s")
            logger.info(f"Train samples/s: {metrics.get('train_samples_per_second', 0):.2f}")
            logger.info("=" * 60)

            return metrics

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM! Reducing batch size and retrying...")
            torch.cuda.empty_cache()
            return self._retry_with_smaller_batch()

    def _retry_with_smaller_batch(self) -> dict:
        """Retry training with smaller batch size after OOM.

        Returns:
            Training metrics dictionary.
        """
        from AuraTrainer.utils.gpu import reduce_batch_size_on_oom

        current_bs = self.config.get("per_device_train_batch_size", 4)
        new_bs = reduce_batch_size_on_oom(self.gpu_config, current_bs)
        self.config["per_device_train_batch_size"] = new_bs
        self.config["gradient_accumulation_steps"] = min(
            self.config.get("gradient_accumulation_steps", 4) * 2, 32
        )

        logger.info(
            f"Retrying with batch_size={new_bs}, "
            f"grad_accum={self.config['gradient_accumulation_steps']}"
        )

        training_args = self._create_training_args()
        data_collator = self._create_data_collator()

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )

        result = self.trainer.train()
        return result.metrics

    def save_model(self, output_dir: str) -> None:
        """Save the trained model.

        Args:
            output_dir: Directory to save model to.
        """
        if self.trainer:
            logger.info(f"Saving model to {output_dir}")
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully")

    def evaluate(self) -> dict:
        """Evaluate the model.

        Returns:
            Evaluation metrics.
        """
        if self.trainer and self.eval_dataset:
            logger.info("Running evaluation...")
            metrics = self.trainer.evaluate()
            logger.info(f"Eval loss: {metrics.get('eval_loss', 'N/A')}")
            return metrics
        return {}
