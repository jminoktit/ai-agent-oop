#!/usr/bin/env python3
"""AuraBook Training Script.

Main entry point for QLoRA fine-tuning TinyLlama-1.1B-Chat
as a university assistant.

Usage:
    python -m AuraTrainer.scripts.train [--config CONFIG] [--size SIZE]
    python -m AuraTrainer.scripts.train --config configs/train.yaml --size 100k
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
import torch
from datasets import Dataset

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from AuraTrainer.utils.logger import setup_logger, get_logger
from AuraTrainer.utils.gpu import auto_configure, Platform
from AuraTrainer.utils.monitor import TrainingMonitor
from AuraTrainer.data.loader import DatasetLoader
from AuraTrainer.data.cleaner import DatasetCleaner
from AuraTrainer.data.formatter import DatasetFormatter
from AuraTrainer.data.deduplicate import DatasetDeduplicator
from AuraTrainer.data.sampler import DatasetSampler
from AuraTrainer.models.loader import ModelLoader
from AuraTrainer.models.lora import LoRAConfigurator
from AuraTrainer.training.trainer import AuraTrainer
from AuraTrainer.training.callbacks import (
    PerformanceCallback,
    MemoryCallback,
    CheckpointCallback,
)
from AuraTrainer.training.checkpoint import CheckpointManager


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_gdrive_colab() -> Optional[str]:
    """Setup Google Drive mount for Colab.

    Returns:
        Google Drive save path or None.
    """
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        gdrive_dir = "/content/drive/MyDrive/AuraBook/checkpoints"
        os.makedirs(gdrive_dir, exist_ok=True)
        logger.info(f"Google Drive mounted at {gdrive_dir}")
        return gdrive_dir
    except (ImportError, Exception):
        return None


def prepare_datasets(
    datasets_config: dict,
    train_config: dict,
    dataset_size: str = "100k",
) -> Dataset:
    """Load, clean, format, and balance all datasets.

    Args:
        datasets_config: Dataset configuration.
        train_config: Training configuration.
        dataset_size: Target dataset size.

    Returns:
        Prepared training Dataset.
    """
    logger = get_logger("AuraTrainer.DataPrep")
    logger.info("=" * 60)
    logger.info(f"Preparing datasets (target: {dataset_size})")
    logger.info("=" * 60)

    cleaner = DatasetCleaner(
        min_text_length=datasets_config.get("cleaning", {}).get("min_text_length", 10),
        max_text_length=datasets_config.get("cleaning", {}).get("max_text_length", 8192),
        min_words=datasets_config.get("cleaning", {}).get("min_words", 3),
        max_words=datasets_config.get("cleaning", {}).get("max_words", 4096),
        remove_empty=datasets_config.get("cleaning", {}).get("remove_empty", True),
        remove_broken_code=datasets_config.get("cleaning", {}).get("remove_broken_code", True),
    )
    formatter = DatasetFormatter()
    deduplicator = DatasetDeduplicator(
        method=datasets_config.get("cleaning", {}).get("dedup_method", "exact")
    )
    loader = DatasetLoader(streaming=True)

    categories = datasets_config.get("categories", {})
    all_data = {}

    for category_name, category_config in categories.items():
        logger.info(f"\n--- Loading category: {category_name} ---")
        category_examples = []

        for ds_config in category_config.get("datasets", []):
            try:
                ds = loader.load_single(
                    name=ds_config["name"],
                    repo=ds_config["repo"],
                    subset=ds_config.get("subset"),
                    split=ds_config.get("split", "train"),
                    streaming=ds_config.get("stream", True),
                )

                fields = ds_config.get("fields", {})
                count = 0
                max_per_ds = 50000

                for example in ds:
                    extracted = {}
                    for target_name, source_name in fields.items():
                        value = example.get(source_name, "")
                        if isinstance(value, list):
                            value = str(value)
                        elif value is None:
                            value = ""
                        extracted[target_name] = str(value)

                    cleaned = cleaner.clean_example(extracted)
                    if cleaned is not None:
                        category_examples.append(cleaned)
                        count += 1

                    if count >= max_per_ds:
                        break

                logger.info(
                    f"  Loaded {count} from {ds_config['name']}"
                )

            except Exception as e:
                logger.warning(f"  Failed to load {ds_config.get('name')}: {e}")
                continue

        if category_examples:
            deduped = deduplicator.deduplicate_batch(category_examples)
            formatted = [
                formatter.format_by_category(ex, category_name)
                for ex in deduped
            ]
            all_data[category_name] = formatted
            logger.info(
                f"Category {category_name}: {len(formatted)} examples after cleaning"
            )

    sampler = DatasetSampler(
        ratios={cat: cfg.get("ratio", 0.2) for cat, cfg in categories.items()},
        seed=42,
        shuffle=True,
    )

    available = {cat: len(texts) for cat, texts in all_data.items()}
    total_desired = DatasetSampler.SIZE_PRESETS.get(dataset_size, 100_000)
    counts = sampler.calculate_counts(total_desired, available)

    balanced_texts = []
    for category, count in counts.items():
        texts = all_data.get(category, [])
        sampled = sampler.sample_dataset(
            [{"text": t} for t in texts], count, category
        )
        for s in sampled:
            balanced_texts.append(s["text"])

    if sampler.shuffle:
        import random
        rng = random.Random(42)
        rng.shuffle(balanced_texts)

    texts = balanced_texts

    dataset = Dataset.from_dict({"text": texts})

    logger.info(f"\nFinal dataset: {len(dataset):,} examples")
    return dataset


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="AuraBook QLoRA Training")
    parser.add_argument(
        "--config", default="configs/train.yaml", help="Training config path"
    )
    parser.add_argument(
        "--datasets-config", default="configs/datasets.yaml", help="Datasets config path"
    )
    parser.add_argument(
        "--size",
        default="100k",
        choices=["50k", "100k", "250k", "500k", "1M", "2M"],
        help="Dataset size",
    )
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    setup_logger("AuraTrainer", log_file=f"train_{int(time.time())}.log")

    global logger
    logger = get_logger("AuraTrainer.Main")

    logger.info("=" * 60)
    logger.info("AuraBook QLoRA Training Pipeline")
    logger.info("=" * 60)

    train_config = load_config(args.config)
    datasets_config = load_config(args.datasets_config)

    gpu_config = auto_configure()

    gdrive_dir = None
    if gpu_config.platform == Platform.GOOGLE_COLAB:
        gdrive_dir = setup_gdrive_colab()

    training = train_config.get("training", {})
    training["per_device_train_batch_size"] = gpu_config.per_device_batch_size
    training["gradient_accumulation_steps"] = gpu_config.gradient_accumulation_steps
    training["fp16"] = gpu_config.fp16
    training["bf16"] = gpu_config.bf16
    training["tf32"] = gpu_config.tf32

    if args.output_dir:
        training["output_dir"] = args.output_dir
    elif gdrive_dir:
        training["output_dir"] = gdrive_dir

    training["gdrive"] = {
        "enabled": gdrive_dir is not None,
        "save_dir": gdrive_dir or "",
    }

    train_dataset = prepare_datasets(datasets_config, training, args.size)

    model_config = train_config.get("model", {})
    qlora_config = train_config.get("qlora", {})

    model_loader = ModelLoader(
        model_name=model_config.get("name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        max_length=model_config.get("max_length", 2048),
    )

    compute_dtype = torch.bfloat16 if gpu_config.bf16 else torch.float16

    model, tokenizer = model_loader.load_for_training(
        quantize=qlora_config.get("enabled", True),
        bits=qlora_config.get("bits", 4),
        compute_dtype=compute_dtype,
    )

    if gpu_config.supports_flash_attention:
        try:
            model = model_loader.enable_gradient_checkpointing(model)
        except Exception:
            pass

    lora_configurator = LoRAConfigurator(
        r=qlora_config.get("lora_r", 64),
        alpha=qlora_config.get("lora_alpha", 128),
        dropout=qlora_config.get("lora_dropout", 0.05),
        bias=qlora_config.get("lora_bias", "none"),
        target_modules=qlora_config.get("lora_target_modules", None),
    )

    model = lora_configurator.apply_to_model(
        model,
        gradient_checkpointing=training.get("gradient_checkpointing", True),
    )

    checkpoint_config = train_config.get("checkpoint", {})
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=training.get("output_dir", "./outputs") + "/checkpoints",
        save_optimizer=checkpoint_config.get("save_optimizer", True),
    )

    resume_path = args.resume
    if not resume_path and training.get("resume_from_checkpoint", False):
        resume_path = checkpoint_mgr.find_latest_checkpoint()

    trainer = AuraTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        gpu_config=gpu_config,
        training_config=training,
    )

    try:
        metrics = trainer.train(resume_from_checkpoint=resume_path)

        output_dir = training.get("output_dir", "./outputs")
        trainer.save_model(os.path.join(output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

        checkpoint_mgr.export_merged_model(
            model,
            os.path.join(output_dir, "merged_model"),
        )

        lora_adapter_dir = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(lora_adapter_dir)

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
