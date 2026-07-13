#!/usr/bin/env python3
"""LoRA Model Merger Script.

Merge trained LoRA adapter with base model.

Usage:
    python -m AuraTrainer.scripts.merge \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter-path ./outputs/lora_adapter \
        --output-dir ./outputs/merged \
        --export-gguf
"""

import argparse
import os
import sys
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from AuraTrainer.utils.logger import setup_logger, get_logger
from AuraTrainer.models.loader import ModelLoader
from AuraTrainer.models.lora import LoRAConfigurator
from AuraTrainer.training.checkpoint import CheckpointManager


def merge_models(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    export_gguf: bool = False,
    export_safetensors: bool = True,
) -> None:
    """Merge LoRA adapter with base model.

    Args:
        base_model_name: HuggingFace base model name.
        adapter_path: Path to trained LoRA adapter.
        output_dir: Output directory for merged model.
        export_gguf: Export to GGUF format.
        export_safetensors: Use safetensors format.
    """
    logger = get_logger("AuraTrainer.Merge")

    logger.info("=" * 60)
    logger.info("Model Merge Pipeline")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_dir}")

    model_loader = ModelLoader(model_name=base_model_name)

    logger.info("Loading base model (non-quantized)...")
    base_model = model_loader.load_model(
        quantize=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info("Loading LoRA adapter...")
    merged_model = LoRAConfigurator.load_adapter(base_model, adapter_path)

    logger.info("Merging weights...")
    final_model = merged_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving merged model to {output_dir}")
    final_model.save_pretrained(
        output_dir, safe_serialization=export_safetensors
    )

    tokenizer = model_loader.load_tokenizer()
    tokenizer.save_pretrained(output_dir)

    logger.info("Merged model saved successfully!")

    checkpoint_mgr = CheckpointManager()
    checkpoint_mgr.export_merged_model(merged_model, output_dir, export_safetensors)

    if export_gguf:
        try:
            export_to_gguf(output_dir)
        except Exception as e:
            logger.warning(f"GGUF export failed: {e}")

    logger.info("=" * 60)
    logger.info("Merge completed!")
    logger.info(f"Merged model: {output_dir}")
    logger.info("=" * 60)


def export_to_gguf(model_dir: str) -> None:
    """Export model to GGUF format using llama.cpp.

    Args:
        model_dir: Directory containing the model.
    """
    logger = get_logger("AuraTrainer.GGUF")

    logger.info("Attempting GGUF export...")
    logger.info("Note: Requires llama.cpp to be installed")

    try:
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "llama_cpp.llama_cpp",
                "convert",
                model_dir,
                "--outfile",
                f"{model_dir}/model.gguf",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            logger.info(f"GGUF exported: {model_dir}/model.gguf")
        else:
            logger.warning(f"GGUF export returned non-zero: {result.stderr}")
    except FileNotFoundError:
        logger.warning(
            "llama.cpp not found. Install it to enable GGUF export.\n"
            "  pip install llama-cpp-python\n"
            "  OR clone and build llama.cpp from source"
        )
    except subprocess.TimeoutExpired:
        logger.warning("GGUF export timed out")


def main():
    """Main merge entry point."""
    parser = argparse.ArgumentParser(description="Merge LoRA Adapter with Base Model")
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name/path",
    )
    parser.add_argument(
        "--adapter-path", required=True, help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output-dir", default="./outputs/merged", help="Output directory"
    )
    parser.add_argument(
        "--export-gguf", action="store_true", help="Export to GGUF format"
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Don't use safetensors format",
    )
    args = parser.parse_args()

    setup_logger("AuraTrainer")

    merge_models(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        export_gguf=args.export_gguf,
        export_safetensors=not args.no_safetensors,
    )


if __name__ == "__main__":
    main()
