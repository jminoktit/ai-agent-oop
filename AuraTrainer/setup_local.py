#!/usr/bin/env python3
"""Local Training Setup Script.

Run this once to setup the environment, then use train.py for training.

Usage:
    python setup_local.py
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def check_python():
    """Check Python version."""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"❌ Python {v.major}.{v.minor} detected. Need Python 3.10+")
        sys.exit(1)
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {name} ({vram:.1f} GB)")
            return True
        else:
            print("❌ No GPU detected!")
            print("   Install CUDA: https://developer.nvidia.com/cuda-downloads")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet (will install)")
        return True


def check_cuda():
    """Check CUDA toolkit."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    print(f"✅ CUDA: {line.strip()}")
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("⚠️  CUDA toolkit not found in PATH")
    print("   Install: https://developer.nvidia.com/cuda-downloads")
    return False


def install_packages():
    """Install required packages."""
    print("\n📦 Installing packages...")

    packages = [
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.4",
        "accelerate>=0.25.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.19.0",
    ]

    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
        )

    # Flash Attention (optional)
    print("   Installing Flash Attention (optional)...")
    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "flash-attn>=2.3.0", "--no-build-isolation",
            ],
            stdout=subprocess.DEVNULL,
            timeout=300,
        )
        print("   ✅ Flash Attention installed")
    except Exception:
        print("   ⚠️  Flash Attention failed (using SDPA fallback)")

    print("✅ All packages installed!")


def create_directories():
    """Create necessary directories."""
    dirs = [
        "outputs",
        "outputs/checkpoints",
        "outputs/final_model",
        "outputs/merged_model",
        "outputs/lora_adapter",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")


def test_imports():
    """Test that all imports work."""
    print("\n🔍 Testing imports...")
    errors = []

    imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("trl", "TRL"),
        ("accelerate", "Accelerate"),
        ("yaml", "PyYAML"),
        ("safetensors", "Safetensors"),
    ]

    for module, name in imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            errors.append(name)

    if errors:
        print(f"\n❌ Missing: {', '.join(errors)}")
        return False

    print("✅ All imports OK!")
    return True


def show_system_info():
    """Show system information."""
    print("\n📊 System Info:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   Machine: {platform.machine()}")

    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
        print(f"   RAM: {ram:.1f} GB")
    except ImportError:
        pass


def main():
    """Main setup function."""
    print("=" * 60)
    print("  AuraBook Local Training Setup")
    print("=" * 60)

    show_system_info()

    print("\n1️⃣  Checking prerequisites...")
    check_python()
    check_cuda()
    check_gpu()

    print("\n2️⃣  Installing packages...")
    install_packages()

    print("\n3️⃣  Creating directories...")
    create_directories()

    print("\n4️⃣  Testing imports...")
    if not test_imports():
        print("\n❌ Setup incomplete. Fix errors above.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ Setup Complete!")
    print("=" * 60)
    print("\n🚀 Start training:")
    print("   python -m AuraTrainer.scripts.train --size 100k")
    print("\n📋 Options:")
    print("   --size       Dataset size: 50k, 100k, 250k, 500k, 1M, 2M")
    print("   --config     Training config file")
    print("   --resume     Resume from checkpoint path")
    print("   --output-dir Override output directory")
    print("\n📋 Other commands:")
    print("   python -m AuraTrainer.scripts.merge --adapter-path ./outputs/lora_adapter")
    print("   python -m AuraTrainer.scripts.inference --model-path ./outputs/merged_model --interactive")


if __name__ == "__main__":
    main()
