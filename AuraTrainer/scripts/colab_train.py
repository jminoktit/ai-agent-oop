#!/usr/bin/env python3
"""Google Colab Training Script for AuraBook QLoRA.

Run this entire script in a single Colab cell, or run cell-by-cell.
Make sure to enable GPU: Runtime → Change runtime type → T4 GPU

Usage:
    1. Open Google Colab
    2. Upload this file or paste contents
    3. Enable GPU runtime
    4. Run All
"""

# ═══════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ═══════════════════════════════════════════════════════════

import subprocess
import sys
import os

# CRITICAL: Change to /content to avoid local module import conflicts
os.chdir('/content')
sys.path = [p for p in sys.path if 'AuraTrainer' not in p and 'ai-agent-op' not in p]

def install_packages():
    """Install required packages for Colab."""
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Flash Attention (optional, may fail on some GPUs)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "flash-attn>=2.3.0",
            "--no-build-isolation"
        ])
        print("✅ Flash Attention installed")
    except Exception:
        print("⚠️  Flash Attention not available (using SDPA fallback)")

install_packages()
print("✅ All packages installed!")


# ═══════════════════════════════════════════════════════════
# CELL 2: Setup Google Drive
# ═══════════════════════════════════════════════════════════

from google.colab import drive

drive.mount("/content/drive")

GDRIVE_BASE = "/content/drive/MyDrive/AuraBook"
CHECKPOINT_DIR = f"{GDRIVE_BASE}/checkpoints"
OUTPUT_DIR = f"{GDRIVE_BASE}/outputs"

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅ Google Drive mounted")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Outputs: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════
# CELL 3: Detect GPU & Auto-Configure
# ═══════════════════════════════════════════════════════════

import torch

def detect_gpu():
    """Detect GPU and return optimal config."""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No GPU detected! Enable GPU: Runtime → Change runtime type")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cc = torch.cuda.get_device_capability(0)

    configs = {
        "Tesla T4": {
            "batch_size": 2, "grad_accum": 8, "bf16": False,
            "fp16": True, "max_length": 1024, "flash_attn": True,
        },
        "NVIDIA L4": {
            "batch_size": 4, "grad_accum": 4, "bf16": True,
            "fp16": False, "max_length": 2048, "flash_attn": True,
        },
        "Tesla V100-SXM2-16GB": {
            "batch_size": 2, "grad_accum": 8, "bf16": False,
            "fp16": True, "max_length": 1024, "flash_attn": True,
        },
        "NVIDIA A100-SXM4-40GB": {
            "batch_size": 8, "grad_accum": 2, "bf16": True,
            "fp16": False, "max_length": 2048, "flash_attn": True,
        },
        "NVIDIA A100-SXM4-80GB": {
            "batch_size": 16, "grad_accum": 1, "bf16": True,
            "fp16": False, "max_length": 4096, "flash_attn": True,
        },
        "NVIDIA GeForce RTX 3060": {
            "batch_size": 2, "grad_accum": 8, "bf16": True,
            "fp16": False, "max_length": 1024, "flash_attn": True,
        },
        "NVIDIA GeForce RTX 4060": {
            "batch_size": 1, "grad_accum": 16, "bf16": True,
            "fp16": False, "max_length": 1024, "flash_attn": True,
        },
        "NVIDIA GeForce RTX 4090": {
            "batch_size": 8, "grad_accum": 2, "bf16": True,
            "fp16": False, "max_length": 2048, "flash_attn": True,
        },
    }

    gpu_config = configs.get(gpu_name, {
        "batch_size": 2, "grad_accum": 8, "bf16": cc >= (8, 0),
        "fp16": cc < (8, 0), "max_length": 1024, "flash_attn": cc >= (8, 0),
    })

    # Flash Attention check
    try:
        import flash_attn
        gpu_config["flash_attn"] = True
        print("✅ Flash Attention available")
    except ImportError:
        print("⚠️  Flash Attention not available (using SDPA)")

    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {vram_gb:.1f} GB")
    print(f"🔧 Compute Capability: {cc[0]}.{cc[1]}")
    print(f"📦 Batch Size: {gpu_config['batch_size']}")
    print(f"📊 Grad Accum: {gpu_config['grad_accum']}")
    print(f"⚡ Effective Batch: {gpu_config['batch_size'] * gpu_config['grad_accum']}")
    print(f"🔀 BF16: {gpu_config['bf16']}, FP16: {gpu_config['fp16']}")

    return gpu_config

gpu_config = detect_gpu()


# ═══════════════════════════════════════════════════════════
# CELL 4: Load Datasets (Streaming)
# ═══════════════════════════════════════════════════════════

from datasets import load_dataset, Dataset
import hashlib
import random

def load_all_datasets():
    """Load datasets with streaming to save RAM."""

    datasets_config = {
        "programming": {
            "ratio": 0.40,
            "datasets": [
                ("HuggingFaceTB/evol-codealpaca", None, "train",
                 {"instruction": "instruction", "input": "input", "output": "output"}),
                ("google-research-datasets/mbpp", "full", "train",
                 {"problem": "question", "solution": "answer"}),
                ("tatsu-lab/alpaca", None, "train",
                 {"instruction": "instruction", "input": "input", "output": "output"}),
                ("code_search_net", "python", "train",
                 {"func_documentation_string": "question", "func_code_string": "answer"}),
            ],
        },
        "chat": {
            "ratio": 0.25,
            "datasets": [
                ("Open-Orca/OpenOrca-Platypus2", None, "train",
                 {"instruction": "instruction", "input": "input", "output": "output"}),
                ("stingning/ultrachat", None, "train",
                 {"prompt": "prompt", "response": "response"}),
                ("OpenAssistant/oasst2", None, "train",
                 {"text": "text", "role": "role"}),
            ],
        },
        "math": {
            "ratio": 0.15,
            "datasets": [
                ("meta-math/MetaMathQA", None, "train",
                 {"query": "question", "response": "answer"}),
                ("openai/gsm8k", "main", "train",
                 {"question": "question", "answer": "answer"}),
            ],
        },
        "education": {
            "ratio": 0.10,
            "datasets": [
                ("openwebtext", None, "train",
                 {"text": "text"}),
            ],
        },
        "arabic": {
            "ratio": 0.10,
            "datasets": [
                ("MBZUAI-AILab/arabic-openhermes", None, "train",
                 {"instruction": "instruction", "input": "input", "output": "output"}),
                ("CohereForMaya/aya_collection", "ar", "train",
                 {"input": "input", "output": "output"}),
            ],
        },
    }

    return datasets_config

DATASETS_CONFIG = load_all_datasets()
print("✅ Dataset config loaded")


# ═══════════════════════════════════════════════════════════
# CELL 5: Clean & Format Data
# ═══════════════════════════════════════════════════════════

import re

SYSTEM_PROMPT = (
    "You are Aura, a helpful university assistant specializing in programming, "
    "mathematics, and education. You can explain programming concepts, write code, "
    "debug errors, explain math problems, and help with studying. You speak both "
    "Arabic and English fluently."
)

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text

def is_valid(text, min_len=10, max_len=8192):
    """Check if text is valid."""
    if not text or len(text) < min_len or len(text) > max_len:
        return False
    words = text.split()
    return 3 <= len(words) <= 4096

def format_chat(user_msg, assistant_msg):
    """Format as TinyLlama chat."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n{assistant_msg}</s>"
    )

def format_code(instruction, input_text, output):
    """Format code instruction."""
    user_msg = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
    return format_chat(user_msg, output)

def format_math(question, answer):
    """Format math problem."""
    user_msg = f"Solve the following math problem step by step:\n{question}"
    return format_chat(user_msg, answer)

# Hash for deduplication
seen_hashes = set()

def compute_hash(text):
    """Compute dedup hash."""
    normalized = " ".join(text.lower().strip().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def is_duplicate(text):
    """Check if text is duplicate."""
    h = compute_hash(text)
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False

print("✅ Cleaning functions ready")


# ═══════════════════════════════════════════════════════════
# CELL 6: Load & Process Data
# ═══════════════════════════════════════════════════════════

DATASET_SIZE = "100k"  # Options: 50k, 100k, 250k, 500k, 1M
SIZE_MAP = {"50k": 50000, "100k": 100000, "250k": 250000, "500k": 500000, "1M": 1000000}
TARGET_TOTAL = SIZE_MAP.get(DATASET_SIZE, 100000)

all_examples = {cat: [] for cat in DATASETS_CONFIG}

for category, cat_config in DATASETS_CONFIG.items():
    ratio = cat_config["ratio"]
    target_per_cat = int(TARGET_TOTAL * ratio)

    print(f"\n📥 Loading {category} (target: {target_per_cat:,})...")

    for repo, subset, split, fields in cat_config["datasets"]:
        try:
            ds = load_dataset(repo, subset, split=split, streaming=True, trust_remote_code=True)

            count = 0
            max_per_ds = target_per_cat // len(cat_config["datasets"]) + 1000

            for example in ds:
                if count >= max_per_ds:
                    break

                # Extract fields
                texts = {}
                for target_name, source_name in fields.items():
                    value = example.get(source_name, "")
                    if isinstance(value, list):
                        value = str(value)
                    texts[target_name] = str(value) if value else ""

                # Get main text for cleaning/dedup
                main_text = texts.get("output", texts.get("answer", texts.get("response", "")))
                main_text = clean_text(main_text)

                if not is_valid(main_text):
                    continue
                if is_duplicate(main_text):
                    continue

                # Format based on category
                if category == "programming":
                    formatted = format_code(
                        texts.get("instruction", texts.get("question", "")),
                        texts.get("input", ""),
                        main_text,
                    )
                elif category == "math":
                    formatted = format_math(
                        texts.get("query", texts.get("question", "")),
                        main_text,
                    )
                else:
                    user_msg = texts.get("instruction", texts.get("prompt", texts.get("input", texts.get("text", ""))))
                    formatted = format_chat(clean_text(user_msg), main_text)

                all_examples[category].append(formatted)
                count += 1

            print(f"  ✅ {repo.split('/')[-1]}: {count} examples")

        except Exception as e:
            print(f"  ⚠️  Failed {repo}: {e}")
            continue

# Balance datasets
print(f"\n⚖️  Balancing datasets (total target: {TARGET_TOTAL:,})...")
balanced = []
for category, examples in all_examples.items():
    ratio = DATASETS_CONFIG[category]["ratio"]
    count = min(int(TARGET_TOTAL * ratio), len(examples))
    sampled = random.sample(examples, count) if count < len(examples) else examples
    balanced.extend(sampled)
    print(f"  {category}: {len(sampled):,} (ratio: {ratio:.0%})")

random.shuffle(balanced)

# Create dataset
dataset = Dataset.from_dict({"text": balanced})

# Split
split_idx = int(len(dataset) * 0.95)
train_dataset = dataset.select(range(split_idx))
eval_dataset = dataset.select(range(split_idx, len(dataset)))

print(f"\n✅ Dataset ready!")
print(f"   Train: {len(train_dataset):,} examples")
print(f"   Eval: {len(eval_dataset):,} examples")


# ═══════════════════════════════════════════════════════════
# CELL 7: Load Model & Apply QLoRA
# ═══════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"📥 Loading model: {MODEL_NAME}")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if gpu_config["bf16"] else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Prepare for training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded!")
print(f"   Total params: {total:,}")
print(f"   Trainable: {trainable:,} ({trainable/total*100:.2f}%)")


# ═══════════════════════════════════════════════════════════
# CELL 8: Training
# ═══════════════════════════════════════════════════════════

NUM_EPOCHS = 3

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=gpu_config["batch_size"],
    gradient_accumulation_steps=gpu_config["grad_accum"],
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    warmup_steps=100,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    fp16=gpu_config["fp16"],
    bf16=gpu_config["bf16"],
    tf32=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=200,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",
    run_name="aura-book-qlora",
    seed=42,
    max_seq_length=gpu_config["max_length"],
    dataset_text_field="text",
    packing=False,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Calculate total steps
total_steps = (
    len(train_dataset)
    // training_args.per_device_train_batch_size
    // training_args.gradient_accumulation_steps
    * NUM_EPOCHS
)

print("🚀 Training Configuration:")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Total steps: {total_steps:,}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Grad accum: {training_args.gradient_accumulation_steps}")
print(f"   Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Save every: {training_args.save_steps} steps")

# Check for existing checkpoint
import glob
checkpoints = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
resume_from = None
if checkpoints:
    resume_from = checkpoints[-1]
    print(f"\n📂 Resuming from: {resume_from}")

# Start training!
print("\n🚀 Starting training...")
result = trainer.train(resume_from_checkpoint=resume_from)

print("\n✅ Training completed!")
print(f"   Final loss: {result.metrics.get('train_loss', 'N/A')}")
print(f"   Runtime: {result.metrics.get('train_runtime', 0):.1f}s")
print(f"   Samples/s: {result.metrics.get('train_samples_per_second', 0):.2f}")


# ═══════════════════════════════════════════════════════════
# CELL 9: Save Models
# ═══════════════════════════════════════════════════════════

# Save LoRA adapter
lora_dir = f"{OUTPUT_DIR}/lora_adapter"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print(f"✅ LoRA adapter saved: {lora_dir}")

# Merge and save
print("🔄 Merging LoRA weights...")
merged_model = model.merge_and_unload()
merged_dir = f"{OUTPUT_DIR}/merged_model"
merged_model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print(f"✅ Merged model saved: {merged_dir}")

# Copy to Drive checkpoint folder too
import shutil
backup_dir = f"{CHECKPOINT_DIR}/final"
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
shutil.copytree(merged_dir, backup_dir)
print(f"✅ Backup saved to Drive: {backup_dir}")


# ═══════════════════════════════════════════════════════════
# CELL 10: Test the Model
# ═══════════════════════════════════════════════════════════

from transformers import pipeline

print("📥 Loading merged model for testing...")
pipe = pipeline(
    "text-generation",
    model=merged_dir,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

def ask_aura(prompt, system=SYSTEM_PROMPT):
    """Ask the Aura assistant a question."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(formatted, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    response = output[0]["generated_text"]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    return response.replace("</s>", "").strip()

# Test cases
tests = [
    ("Python", "Write a Python function to check if a number is prime."),
    ("SQL", "Write a SQL query to find duplicate emails in a Users table."),
    ("Math", "Explain the Pythagorean theorem with a simple example."),
    ("Arabic", "اشرح لي الفرق بين Stack و Queue بالعربي"),
    ("English", "What is the time complexity of binary search?"),
]

print("\n" + "=" * 60)
print("  Testing AuraBook Assistant")
print("=" * 60)

for category, prompt in tests:
    print(f"\n📝 {category}: {prompt}")
    print(f"🤖 Aura: {ask_aura(prompt)}")
    print("-" * 60)

print("\n🎉 All tests completed!")
print(f"\n📁 Models saved to Google Drive: {OUTPUT_DIR}")
