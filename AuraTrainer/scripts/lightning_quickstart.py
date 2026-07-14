# Lightning AI - Quick Start
# هذا الملف ترفعه مباشرة على Lightning AI Studio

# 1. روح على https://lightning.ai/studio
# 2. اعمل Studio جديد (GPU T4)
# 3. ارفع هذا الملف
# 4. شغّل: python lightning_quickstart.py

import os
import subprocess
import sys

# ── Install Dependencies ──
print("📦 Installing dependencies...")
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

# ── Config ──
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMAIL = os.getenv("NOTIFY_EMAIL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")

# ── GPU Check ──
import torch
if not torch.cuda.is_available():
    print("❌ No GPU! Go to Settings → Accelerator → GPU")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")

# ── Login to HuggingFace ──
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace!")
else:
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print("⚠️ No HF_TOKEN, using TinyLlama")

# ── Load Model ──
print(f"📥 Loading model: {MODEL_NAME}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model loaded! {model.num_parameters():,} parameters")

# ── Apply LoRA ──
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load Dataset ──
print("📥 Loading datasets...")

from datasets import load_dataset, concatenate_datasets

arabic_data = load_dataset("arbml/alpaca_arabic", split="train")
python_data = load_dataset("code-search-net/code_search_net", "python", split="train")

# Sample to manageable size
arabic_data = arabic_data.shuffle(seed=42).select(range(min(50000, len(arabic_data))))
python_data = python_data.shuffle(seed=42).select(range(min(50000, len(python_data))))

def format_chat(example):
    if "instruction" in example:
        user_msg = example.get("instruction", "")
        assistant_msg = example.get("output", example.get("response", ""))
    elif "code" in example:
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

arabic_data = arabic_data.map(format_chat, remove_columns=arabic_data.column_names)
python_data = python_data.map(format_chat, remove_columns=python_data.column_names)
dataset = concatenate_datasets([arabic_data, python_data]).shuffle(seed=42)

print(f"✅ Dataset: {len(dataset)} samples")

# ── Train ──
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

print("🚀 Starting training...")
result = trainer.train()

print(f"✅ Training completed! Loss: {result.training_loss:.4f}")

# ── Save ──
model.save_pretrained("./checkpoints")
tokenizer.save_pretrained("./checkpoints")
print("✅ Model saved to ./checkpoints")

# ── Email Notification ──
if EMAIL:
    import smtplib
    from email.mime.text import MIMEText

    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if smtp_user and smtp_pass:
        msg = MIMEText(f"Training completed!\nLoss: {result.training_loss:.4f}")
        msg["Subject"] = f"✅ Training Done - {MODEL_NAME}"
        msg["From"] = smtp_user
        msg["To"] = EMAIL

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.send_message(msg)
            print(f"📧 Email sent to {EMAIL}")
        except Exception as e:
            print(f"❌ Email failed: {e}")

print("🎉 Done!")
