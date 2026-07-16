"""Kaggle Training Notebook - AuraTrainer.

This notebook uses the OOP structure from AuraTrainer.
Upload this file to Kaggle as a new notebook.
GPU: Select P100 or T4 x2
"""

# %% [markdown]
# # AuraTrainer - Kaggle Training
# Fine-tune Gemma-2-2B-it with QLoRA on free Kaggle GPU

# %% [markdown]
# ## 1. Setup & Install Dependencies

# %%
import subprocess, sys

# Install required packages
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

print("✅ Dependencies installed!")

# %% [markdown]
# ## 2. Clone Repository

# %%
import os

# Clone the repo
repo_url = "https://github.com/jminoktit/ai-agent-oop.git"
clone_dir = "/kaggle/working/ai-agent-oop"

if not os.path.exists(clone_dir):
    !git clone {repo_url} {clone_dir}
    print("✅ Repository cloned!")
else:
    print("✅ Repository already exists!")

# Change to project directory
os.chdir(clone_dir)
sys.path.insert(0, os.path.dirname(clone_dir))

print(f"📁 Working directory: {os.getcwd()}")

# %% [markdown]
# ## 3. Login to HuggingFace (Required for Gemma)

# %%
from huggingface_hub import login

# Enter your HuggingFace token
HF_TOKEN = ""  # <-- Paste your token here: https://huggingface.co/settings/tokens

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace!")
else:
    print("⚠️ No token provided. Using TinyLlama instead of Gemma.")

# %% [markdown]
# ## 4. Import AuraTrainer OOP Classes

# %%
from AuraTrainer.core import AuraTrainer, TrainingConfig

print("✅ AuraTrainer imported!")

# %% [markdown]
# ## 5. Create Training Configuration

# %%
config = TrainingConfig(
    model_name="google/gemma-2-2b-it" if HF_TOKEN else "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_size=100000,
    num_epochs=3,
    learning_rate=2e-4,
    output_dir="./checkpoints",
    hf_token=HF_TOKEN,
)

print("📋 Configuration created:")
print(f"  Model: {config.model_name}")
print(f"  Dataset size: {config.dataset_size}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Learning rate: {config.learning_rate}")

# %% [markdown]
# ## 6. Run Training Pipeline

# %%
trainer = AuraTrainer(config)
metrics = trainer.run()

# %% [markdown]
# ## 7. View Results

# %%
print("🎯 Training Results:")
print(f"  Loss: {metrics.get('train_loss', 'N/A')}")
print(f"  Runtime: {metrics.get('train_runtime', 0):.1f}s")
print(f"  Samples/s: {metrics.get('train_samples_per_second', 0):.2f}")

# %% [markdown]
# ## 8. Download Model

# %%
import shutil
shutil.make_archive("aura-trainer-gemma-2b", "zip", config.output_dir)

print("📦 Model zipped! Click the Files tab on the left to download.")
