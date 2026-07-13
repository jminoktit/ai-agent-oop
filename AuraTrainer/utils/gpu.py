"""GPU Detection and Auto-Configuration."""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.GPU")


class Platform(Enum):
    """Supported platforms."""

    GOOGLE_COLAB = "google_colab"
    KAGGLE = "kaggle"
    LIGHTNING_AI = "lightning_ai"
    LOCAL_LINUX = "local_linux"
    LOCAL_WINDOWS = "local_windows"
    UNKNOWN = "unknown"


class GPUType(Enum):
    """Supported GPU types with memory and capability info."""

    TESLA_T4 = "Tesla T4"
    L4 = "NVIDIA L4"
    V100 = "Tesla V100-SXM2-16GB"
    A100 = "NVIDIA A100-SXM4-40GB"
    A100_80GB = "NVIDIA A100-SXM4-80GB"
    RTX_3060 = "NVIDIA GeForce RTX 3060"
    RTX_4060 = "NVIDIA GeForce RTX 4060"
    RTX_4090 = "NVIDIA GeForce RTX 4090"
    UNKNOWN = "Unknown GPU"


@dataclass
class GPUConfig:
    """Auto-detected GPU configuration."""

    gpu_type: GPUType = GPUType.UNKNOWN
    gpu_name: str = ""
    vram_gb: float = 0.0
    supports_bf16: bool = False
    supports_fp16: bool = True
    supports_tf32: bool = False
    supports_flash_attention: bool = False
    compute_capability: tuple = (0, 0)
    platform: Platform = Platform.UNKNOWN
    device_count: int = 0

    # Auto-tuned parameters
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    bf16: bool = False
    tf32: bool = False
    max_length: int = 2048
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4


# GPU memory and capability mapping
GPU_SPECS: dict[str, dict] = {
    "Tesla T4": {
        "memory_gb": 16,
        "bf16": False,
        "fp16": True,
        "tf32": False,
        "flash_attn": True,
        "compute_capability": (7, 5),
        "recommended_batch": 2,
        "recommended_grad_accum": 8,
    },
    "NVIDIA L4": {
        "memory_gb": 24,
        "bf16": True,
        "fp16": True,
        "tf32": True,
        "flash_attn": True,
        "compute_capability": (8, 9),
        "recommended_batch": 4,
        "recommended_grad_accum": 4,
    },
    "Tesla V100-SXM2-16GB": {
        "memory_gb": 16,
        "bf16": False,
        "fp16": True,
        "tf32": False,
        "flash_attn": True,
        "compute_capability": (7, 0),
        "recommended_batch": 2,
        "recommended_grad_accum": 8,
    },
    "NVIDIA A100-SXM4-40GB": {
        "memory_gb": 40,
        "bf16": True,
        "fp16": True,
        "tf32": True,
        "flash_attn": True,
        "compute_capability": (8, 0),
        "recommended_batch": 8,
        "recommended_grad_accum": 2,
    },
    "NVIDIA A100-SXM4-80GB": {
        "memory_gb": 80,
        "bf16": True,
        "fp16": True,
        "tf32": True,
        "flash_attn": True,
        "compute_capability": (8, 0),
        "recommended_batch": 16,
        "recommended_grad_accum": 1,
    },
    "NVIDIA GeForce RTX 3060": {
        "memory_gb": 12,
        "bf16": True,
        "fp16": True,
        "tf32": True,
        "flash_attn": True,
        "compute_capability": (8, 6),
        "recommended_batch": 2,
        "recommended_grad_accum": 8,
    },
    "NVIDIA GeForce RTX 4060": {
        "memory_gb": 8,
        "bf16": True,
        "fp16": True,
        "tf32": True,
        "flash_attn": True,
        "compute_capability": (8, 9),
        "recommended_batch": 1,
        "recommended_grad_accum": 16,
    },
    "NVIDIA GeForce RTX 4090": {
        "memory_gb": 24,
        "bf16": True,
        "fp16": True,
        "tf32": True,
        "flash_attn": True,
        "compute_capability": (8, 9),
        "recommended_batch": 8,
        "recommended_grad_accum": 2,
    },
}


def detect_platform() -> Platform:
    """Detect the current running platform.

    Returns:
        Detected platform enum value.
    """
    if os.path.exists("/content"):
        logger.info("Detected platform: Google Colab")
        return Platform.GOOGLE_COLAB
    if os.path.exists("/kaggle"):
        logger.info("Detected platform: Kaggle")
        return Platform.KAGGLE
    if os.environ.get("LIGHTNING_CLOUD"):
        logger.info("Detected platform: Lightning AI")
        return Platform.LIGHTNING_AI
    if platform.system() == "Linux":
        logger.info("Detected platform: Local Linux")
        return Platform.LOCAL_LINUX
    if platform.system() == "Windows":
        logger.info("Detected platform: Local Windows")
        return Platform.LOCAL_WINDOWS
    logger.warning("Detected platform: Unknown")
    return Platform.UNKNOWN


def get_gpu_info() -> Optional[dict]:
    """Get GPU information via nvidia-smi.

    Returns:
        Dictionary with GPU info or None if no GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return None

        parts = lines[0].split(", ")
        return {
            "name": parts[0].strip(),
            "memory_total_mb": float(parts[1].strip()),
            "compute_capability": tuple(
                map(int, parts[2].strip().split("."))
            ),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError, ValueError) as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


def get_vram_usage() -> dict:
    """Get current VRAM usage.

    Returns:
        Dictionary with VRAM usage stats.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

        parts = result.stdout.strip().split(", ")
        return {
            "used_mb": float(parts[0].strip()),
            "total_mb": float(parts[1].strip()),
            "free_mb": float(parts[2].strip()),
        }
    except Exception:
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0}


def check_flash_attention_support() -> bool:
    """Check if flash attention is available.

    Returns:
        True if flash attention is supported.
    """
    try:
        import flash_attn  # noqa: F401

        logger.info("Flash Attention 2 is available")
        return True
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc >= (8, 0):
                logger.info("SDPA Flash Attention is supported")
                return True
    except ImportError:
        pass

    logger.info("Flash Attention not available, using standard attention")
    return False


def auto_configure() -> GPUConfig:
    """Auto-detect GPU and platform, then configure training parameters.

    Returns:
        Auto-configured GPUConfig.
    """
    config = GPUConfig()
    config.platform = detect_platform()

    gpu_info = get_gpu_info()
    if gpu_info is None:
        logger.warning("No GPU detected. Using CPU defaults.")
        config.per_device_batch_size = 1
        config.fp16 = False
        config.bf16 = False
        config.tf32 = False
        return config

    gpu_name = gpu_info["name"]
    config.gpu_name = gpu_name
    config.vram_gb = gpu_info["memory_total_mb"] / 1024
    config.compute_capability = gpu_info["compute_capability"]
    config.device_count = 1

    for gpu_type in GPUType:
        if gpu_type.value == gpu_name:
            config.gpu_type = gpu_type
            break

    specs = GPU_SPECS.get(gpu_name, {})
    if specs:
        config.supports_bf16 = specs.get("bf16", False)
        config.supports_fp16 = specs.get("fp16", True)
        config.supports_tf32 = specs.get("tf32", False)
        config.supports_flash_attention = specs.get("flash_attn", False)
        config.per_device_batch_size = specs.get("recommended_batch", 4)
        config.gradient_accumulation_steps = specs.get("recommended_grad_accum", 4)
    else:
        config.supports_bf16 = gpu_info["compute_capability"] >= (8, 0)
        config.supports_tf32 = gpu_info["compute_capability"] >= (8, 0)

    config.bf16 = config.supports_bf16
    config.fp16 = not config.supports_bf16 and config.supports_fp16
    config.tf32 = config.supports_tf32

    config.supports_flash_attention = check_flash_attention_support()

    if config.vram_gb <= 8:
        config.per_device_batch_size = 1
        config.gradient_accumulation_steps = 16
        config.max_length = 1024
    elif config.vram_gb <= 16:
        config.per_device_batch_size = 2
        config.gradient_accumulation_steps = 8
        config.max_length = 2048
    elif config.vram_gb <= 24:
        config.per_device_batch_size = 4
        config.gradient_accumulation_steps = 4
        config.max_length = 2048
    elif config.vram_gb <= 40:
        config.per_device_batch_size = 8
        config.gradient_accumulation_steps = 2
        config.max_length = 2048
    else:
        config.per_device_batch_size = 16
        config.gradient_accumulation_steps = 1
        config.max_length = 4096

    if config.platform in (Platform.GOOGLE_COLAB, Platform.KAGGLE):
        config.dataloader_num_workers = 2
        config.per_device_batch_size = min(config.per_device_batch_size, 4)

    logger.info(f"GPU: {gpu_name} ({config.vram_gb:.1f} GB)")
    logger.info(f"Platform: {config.platform.value}")
    logger.info(f"Batch size: {config.per_device_batch_size}")
    logger.info(f"Grad accum: {config.gradient_accumulation_steps}")
    logger.info(f"BF16: {config.bf16}, FP16: {config.fp16}, TF32: {config.tf32}")
    logger.info(f"Flash Attention: {config.supports_flash_attention}")

    return config


def reduce_batch_size_on_oom(
    config: GPUConfig, current_batch_size: int
) -> int:
    """Reduce batch size when OOM occurs.

    Args:
        config: Current GPU configuration.
        current_batch_size: Current batch size.

    Returns:
        New reduced batch size.
    """
    new_batch_size = max(1, current_batch_size // 2)
    logger.warning(
        f"OOM detected! Reducing batch size: {current_batch_size} -> {new_batch_size}"
    )
    config.per_device_batch_size = new_batch_size
    config.gradient_accumulation_steps = min(
        config.gradient_accumulation_steps * 2, 32
    )
    return new_batch_size
