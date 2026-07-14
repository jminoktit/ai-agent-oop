# ─────────────────────────────────────────────────────────────
# Lightning AI Training - AuraTrainer
# Free tier: 22 GPU hours/month
# Deploy: lightning run app app_cloud.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import subprocess
import smtplib
from email.mime.text import MIMEText

# Install dependencies
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
], check=True)


def send_email(to_email, subject, body):
    """Send email notification."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        print("⚠️ SMTP not configured, skipping email")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        print(f"📧 Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset, concatenate_datasets

    # ── Config ──
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")
    EMAIL = os.getenv("NOTIFY_EMAIL", "")
    DATASET_SIZE = int(os.getenv("DATASET_SIZE", "100000"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
    GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
    EPOCHS = int(os.getenv("EPOCHS", "3"))
    LR = float(os.getenv("LR", "2e-4"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./checkpoints")

    # Login to HuggingFace
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("✅ Logged in to HuggingFace!")
    else:
        MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print("⚠️ No HF_TOKEN, using TinyLlama")

    # ── GPU Detection ──
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Auto-adjust batch size
        if gpu_mem < 16:
            BATCH_SIZE = 2
            GRAD_ACCUM = 8
        elif gpu_mem < 24:
            BATCH_SIZE = 4
            GRAD_ACCUM = 4
        else:
            BATCH_SIZE = 8
            GRAD_ACCUM = 2
    else:
        print("❌ No GPU! This script requires a GPU.")
        return

    # ── Load Model ──
    print(f"📥 Loading model: {MODEL_NAME}")

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

    arabic_data = load_dataset("arbml/alpaca_arabic", split="train")
    python_data = load_dataset("code-search-net/code_search_net", "python", split="train")

    # Sample
    arabic_data = arabic_data.shuffle(seed=42).select(range(min(DATASET_SIZE // 2, len(arabic_data))))
    python_data = python_data.shuffle(seed=42).select(range(min(DATASET_SIZE // 2, len(python_data))))

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
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
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
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")

    # ── Email Notification ──
    if EMAIL:
        send_email(
            EMAIL,
            f"✅ Training Completed - {MODEL_NAME}",
            f"Training completed successfully!\n\n"
            f"Model: {MODEL_NAME}\n"
            f"Steps: {result.global_step}\n"
            f"Loss: {result.training_loss:.4f}\n"
            f"GPU: {gpu_name}\n"
        )

    print("🎉 Done!")


if __name__ == "__main__":
    main()
