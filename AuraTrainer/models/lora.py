"""LoRA Configuration and Application."""

from typing import List, Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import PreTrainedModel

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.LoRA")


# Default target modules for TinyLlama and similar LLaMA-based models
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class LoRAConfigurator:
    """Configure and apply LoRA adapters."""

    def __init__(
        self,
        r: int = 64,
        alpha: int = 128,
        dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        target_modules: Optional[List[str]] = None,
    ):
        """Initialize LoRA configurator.

        Args:
            r: LoRA rank.
            alpha: LoRA alpha parameter.
            dropout: LoRA dropout.
            bias: Bias training mode.
            task_type: Task type string.
            target_modules: Target module names.
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.task_type = task_type
        self.target_modules = target_modules or DEFAULT_TARGET_MODULES

    def create_config(self) -> LoraConfig:
        """Create PEFT LoRA configuration.

        Returns:
            LoraConfig instance.
        """
        from peft import TaskType

        task_type_enum = TaskType.CAUSAL_LM

        config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=task_type_enum,
            target_modules=self.target_modules,
        )

        logger.info(
            f"LoRA config: r={self.r}, alpha={self.alpha}, "
            f"dropout={self.dropout}, targets={len(self.target_modules)} modules"
        )
        return config

    def apply_to_model(
        self,
        model: PreTrainedModel,
        config: Optional[LoraConfig] = None,
        gradient_checkpointing: bool = True,
    ) -> PeftModel:
        """Apply LoRA adapters to a model.

        Args:
            model: Base model to apply LoRA to.
            config: Optional LoRA config. Creates default if None.
            gradient_checkpointing: Enable gradient checkpointing.

        Returns:
            Model with LoRA adapters applied.
        """
        if config is None:
            config = self.create_config()

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

        peft_model = get_peft_model(model, config)

        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in peft_model.parameters())

        logger.info(
            f"LoRA applied: {trainable_params:,} trainable / "
            f"{total_params:,} total ({trainable_params/total_params*100:.2f}%)"
        )

        peft_model.print_trainable_parameters()
        return peft_model

    @staticmethod
    def merge_and_save(
        peft_model: PeftModel,
        output_dir: str,
        safe_serialization: bool = True,
    ) -> None:
        """Merge LoRA weights into base model and save.

        Args:
            peft_model: Model with LoRA adapters.
            output_dir: Directory to save merged model.
            safe_serialization: Use safetensors format.
        """
        logger.info("Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(
            output_dir, safe_serialization=safe_serialization
        )
        logger.info("Merged model saved successfully")

    @staticmethod
    def save_adapter(
        peft_model: PeftModel,
        output_dir: str,
        safe_serialization: bool = True,
    ) -> None:
        """Save LoRA adapter separately.

        Args:
            peft_model: Model with LoRA adapters.
            output_dir: Directory to save adapter.
            safe_serialization: Use safetensors format.
        """
        logger.info(f"Saving LoRA adapter to {output_dir}")
        peft_model.save_pretrained(
            output_dir, safe_serialization=safe_serialization
        )
        logger.info("LoRA adapter saved successfully")

    @staticmethod
    def load_adapter(
        base_model: PreTrainedModel,
        adapter_path: str,
    ) -> PeftModel:
        """Load a saved LoRA adapter onto a base model.

        Args:
            base_model: Base model to load adapter onto.
            adapter_path: Path to saved adapter.

        Returns:
            Model with adapter loaded.
        """
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("LoRA adapter loaded successfully")
        return model
