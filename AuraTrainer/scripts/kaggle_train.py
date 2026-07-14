# ─────────────────────────────────────────────────────────────
# Kaggle Training Notebook - AuraTrainer
# Upload this file to Kaggle as a new notebook
# GPU: Select P100 or T4 x2
# ─────────────────────────────────────────────────────────────

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
    "yaml",
    "huggingface_hub",
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
# ## 4. Detect GPU

# %%
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_mem:.1f} GB")
    print(f"🔧 CUDA: {torch.version.cuda}")
else:
    print("❌ No GPU detected! Go to Runtime > Change runtime type > GPU")
    print("   Kaggle: Settings > Accelerator > GPU")

# %% [markdown]
# ## 5. Load Configuration

# %%
import yaml

# Auto-configure based on GPU
config = {
    "model": {
        "name": "google/gemma-2-2b-it" if HF_TOKEN else "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_seq_length": 1024,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "training": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
    },
}

# Adjust batch size based on GPU memory
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    if gpu_mem < 16:  # T4
        config["training"]["per_device_train_batch_size"] = 2
        config["training"]["gradient_accumulation_steps"] = 8
    elif gpu_mem < 24:  # P100
        config["training"]["per_device_train_batch_size"] = 4
        config["training"]["gradient_accumulation_steps"] = 4
    else:  # V100, A100
        config["training"]["per_device_train_batch_size"] = 8
        config["training"]["gradient_accumulation_steps"] = 2

print("📋 Configuration:")
print(f"  Model: {config['model']['name']}")
print(f"  Batch size: {config['training']['per_device_train_batch_size']}")
print(f"  Gradient accum: {config['training']['gradient_accumulation_steps']}")
print(f"  Effective batch: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
print(f"  bf16: {config['training']['bf16']}")

# %% [markdown]
# ## 6. Load Model with 4-bit Quantization

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = config["model"]["name"]

print(f"📥 Loading model: {model_name}")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model loaded! Parameters: {model.num_parameters():,}")

# %% [markdown]
# ## 7. Apply LoRA

# %%
# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    target_modules=config["lora"]["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 8. Load Dataset

# %%
from datasets import load_dataset

print("📥 Loading datasets...")

# Load Arabic dataset
arabic_data = load_dataset("arbml/alpaca_arabic", split="train")
print(f"  Arabic: {len(arabic_data)} samples")

# Load Python dataset
python_data = load_dataset("code-search-net/code_search_net", "python", split="train")
print(f"  Python: {len(python_data)} samples")

# Sample to manageable size
arabic_data = arabic_data.shuffle(seed=42).select(range(min(50000, len(arabic_data))))
python_data = python_data.shuffle(seed=42).select(range(min(50000, len(python_data))))

print(f"✅ Loaded {len(arabic_data) + len(python_data)} samples total")

# %% [markdown]
# ## 9. Format Dataset

# %%
def format_chat(example):
    """Format data for chat template."""
    if "instruction" in example:
        # Arabic Alpaca format
        user_msg = example.get("instruction", "")
        assistant_msg = example.get("output", "")
        if not assistant_msg:
            assistant_msg = example.get("response", "")
    elif "code" in example:
        # CodeSearchNet format
        user_msg = f"Write Python code for: {example.get('repo_name', '')} {example.get('func_name', '')}"
        assistant_msg = example.get("code", "")
    else:
        user_msg = example.get("prompt", example.get("input", ""))
        assistant_msg = example.get("response", example.get("output", ""))

    if not user_msg or not assistant_msg:
        return {"text": ""}

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# Format datasets
arabic_data = arabic_data.map(format_chat, remove_columns=arabic_data.column_names)
python_data = python_data.map(format_chat, remove_columns=python_data.column_names)

# Merge datasets
from datasets import concatenate_datasets
dataset = concatenate_datasets([arabic_data, python_data]).shuffle(seed=42)

print(f"✅ Dataset formatted: {len(dataset)} samples")

# %% [markdown]
# ## 10. Train with SFTTrainer

# %%
from trl import SFTTrainer
from transformers import TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    num_train_epochs=config["training"]["num_train_epochs"],
    learning_rate=config["training"]["learning_rate"],
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    save_steps=config["training"]["save_steps"],
    save_total_limit=3,
    bf16=config["training"]["bf16"],
    fp16=config["training"]["fp16"],
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    report_to="none",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config["model"]["max_seq_length"],
    args=training_args,
)

print("🚀 Starting training...")
print(f"  Epochs: {config['training']['num_train_epochs']}")
print(f"  Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")

# %%
# Train!
train_result = trainer.train()

print("✅ Training completed!")
print(f"  Total steps: {train_result.global_step}")
print(f"  Final loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ## 11. Save Model

# %%
# Save LoRA adapter
output_dir = "./aura-trainer-gemma-2b"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to {output_dir}")

# %% [markdown]
# ## 12. Test Inference

# %%
from transformers import pipeline

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test
test_prompts = [
    "Write a Python function to sort a list",
    "اكتب دالة تحسب مجموع الأرقام",
    "What is machine learning?",
]

print("🧪 Testing inference:")
for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    output = pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.7)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {output[0]['generated_text'][-1]['content'][:200]}")

# %% [markdown]
# ## 13. Download Model

# %%
# Create a zip for download
import shutil
shutil.make_archive("aura-trainer-gemma-2b", "zip", output_dir)

print("📦 Model zipped! Click the Files tab on the left to download.")
print(f"📁 File: /kaggle/working/ai-agent-oop/aura-trainer-gemma-2b.zip")

# %% [markdown]
# ## 14. Upload to HuggingFace Hub (Optional)

# %%
# Uncomment to upload your model to HuggingFace Hub
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(
#     folder_path=output_dir,
#     repo_id="YOUR_USERNAME/aura-trainer-gemma-2b",
#     repo_type="model",
# )
# print("✅ Uploaded to HuggingFace Hub!")
