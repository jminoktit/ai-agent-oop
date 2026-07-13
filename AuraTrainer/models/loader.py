"""Model Loading with Quantization Support."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.ModelLoader")


class ModelLoader:
    """Load models with optional 4-bit quantization."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_length: int = 2048,
        trust_remote_code: bool = True,
    ):
        """Initialize model loader.

        Args:
            model_name: HuggingFace model name/path.
            max_length: Maximum sequence length.
            trust_remote_code: Trust remote code for custom models.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer.

        Returns:
            Loaded tokenizer.
        """
        logger.info(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token")

        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|system|>\n{{ message['content'] }}</s>\n"
                "{% elif message['role'] == 'user' %}"
                "<|user|>\n{{ message['content'] }}</s>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|assistant|>\n{{ message['content'] }}</s>\n"
                "{% endif %}"
                "{% endfor %}"
            )
            logger.info("Set default chat template")

        logger.info(
            f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
            f"max_length={tokenizer.model_max_length}"
        )
        return tokenizer

    def load_model(
        self,
        quantize: bool = True,
        bits: int = 4,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ) -> PreTrainedModel:
        """Load model with optional quantization.

        Args:
            quantize: Whether to use 4-bit quantization.
            bits: Quantization bits (4 or 8).
            compute_dtype: Compute dtype for quantization.
            double_quant: Use nested quantization.
            quant_type: Quantization type (nf4 or fp4).
            device_map: Device mapping strategy.
            torch_dtype: Model dtype when not quantized.

        Returns:
            Loaded model.
        """
        logger.info(f"Loading model: {self.model_name}")

        bnb_config = None
        if quantize:
            logger.info(
                f"Quantization: {bits}-bit {quant_type}, "
                f"compute_dtype={compute_dtype}, "
                f"double_quant={double_quant}"
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=double_quant,
            )

        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "device_map": device_map,
            "quantization_config": bnb_config,
        }

        if not quantize:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        if not quantize:
            model.config.use_cache = False

        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model loaded: {param_count:,} total params, "
            f"{trainable_count:,} trainable"
        )

        return model

    def load_for_training(
        self,
        quantize: bool = True,
        bits: int = 4,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer for training.

        Args:
            quantize: Whether to use 4-bit quantization.
            bits: Quantization bits.
            compute_dtype: Compute dtype for quantization.

        Returns:
            Tuple of (model, tokenizer).
        """
        tokenizer = self.load_tokenizer()
        model = self.load_model(
            quantize=quantize,
            bits=bits,
            compute_dtype=compute_dtype,
        )
        return model, tokenizer

    @staticmethod
    def enable_gradient_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
        """Enable gradient checkpointing on model.

        Args:
            model: The model to enable checkpointing on.

        Returns:
            Model with gradient checkpointing enabled.
        """
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled")
        return model

    @staticmethod
    def get_model_info(model: PreTrainedModel) -> dict:
        """Get model information.

        Args:
            model: The model to inspect.

        Returns:
            Dictionary with model info.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        return {
            "model_name": getattr(model.config, "_name_or_path", "unknown"),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": (trainable_params / total_params * 100)
            if total_params > 0
            else 0,
            "vocab_size": getattr(model.config, "vocab_size", 0),
            "hidden_size": getattr(model.config, "hidden_size", 0),
            "num_layers": getattr(model.config, "num_hidden_layers", 0),
            "max_position_embeddings": getattr(
                model.config, "max_position_embeddings", 0
            ),
        }
