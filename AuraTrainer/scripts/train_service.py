#!/usr/bin/env python3
"""
AuraBook Backend Training Service.
Trains the model and sends email notification when complete.

Usage:
    python train_service.py --email your@email.com --size 100k
"""

import argparse
import os
import sys
import json
import time
import smtplib
import hashlib
import random
import re
import gc
import glob
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are Aura, a helpful university assistant specializing in programming, "
    "mathematics, and education. You can explain programming concepts, write code, "
    "debug errors, explain math problems, and help with studying. You speak both "
    "Arabic and English fluently."
)

DATASETS = {
    "programming": {
        "ratio": 0.40,
        "sources": [
            ("tatsu-lab/alpaca", None, "train",
             {"instruction": "instruction", "input": "input", "output": "output"}),
            ("code-search-net/code_search_net", "python", "train",
             {"func_documentation_string": "question", "func_code_string": "answer"}),
        ],
    },
    "chat": {
        "ratio": 0.25,
        "sources": [
            ("Open-Orca/OpenOrca", None, "train",
             {"instruction": "instruction", "input": "input", "output": "output"}),
            ("databricks/databricks-dolly-15k", None, "train",
             {"instruction": "instruction", "context": "context", "response": "response"}),
        ],
    },
    "math": {
        "ratio": 0.15,
        "sources": [
            ("meta-math/MetaMathQA", None, "train",
             {"query": "query", "response": "response"}),
            ("openai/gsm8k", "main", "train",
             {"question": "question", "answer": "answer"}),
        ],
    },
    "education": {
        "ratio": 0.10,
        "sources": [
            ("databricks/databricks-dolly-15k", None, "train",
             {"instruction": "instruction", "context": "context", "response": "response"}),
        ],
    },
    "arabic": {
        "ratio": 0.10,
        "sources": [
            ("arbml/alpaca_arabic", None, "train",
             {"instruction": "instruction", "input": "input", "output": "output"}),
        ],
    },
}


# ═══════════════════════════════════════════════════════════
# EMAIL NOTIFICATION
# ═══════════════════════════════════════════════════════════

def send_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    sender_email: str = None,
    sender_password: str = None,
) -> bool:
    """Send email notification.

    Args:
        to_email: Recipient email.
        subject: Email subject.
        body: Email body.
        smtp_server: SMTP server.
        smtp_port: SMTP port.
        sender_email: Sender email (from env).
        sender_password: Sender password/app password (from env).

    Returns:
        True if sent successfully.
    """
    sender_email = sender_email or os.environ.get("SMTP_EMAIL")
    sender_password = sender_password or os.environ.get("SMTP_PASSWORD")

    if not sender_email or not sender_password:
        print("Warning: SMTP not configured, skipping email")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


def notify_start(to_email: str, config: dict):
    """Send training started notification."""
    subject = "🚀 AuraBook Training Started"
    body = f"""
    <h2>Training Started</h2>
    <p><strong>Model:</strong> {config['model_name']}</p>
    <p><strong>Samples:</strong> {config['total_samples']:,}</p>
    <p><strong>Rounds:</strong> {config['num_rounds']}</p>
    <p><strong>Started at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
    <p>You'll receive another email when training completes.</p>
    """
    send_email(to_email, subject, body)


def notify_complete(to_email: str, results: dict):
    """Send training completed notification."""
    subject = "✅ AuraBook Training Complete!"
    body = f"""
    <h2>Training Complete!</h2>
    <p><strong>Status:</strong> Success</p>
    <p><strong>Total rounds:</strong> {results['total_rounds']}</p>
    <p><strong>Final loss:</strong> {results['final_loss']:.4f}</p>
    <p><strong>Total time:</strong> {results['total_time']:.1f} seconds</p>
    <p><strong>Completed at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
    <h3>Models saved:</h3>
    <ul>
        <li>Merged model: {results['merged_dir']}</li>
        <li>LoRA adapter: {results['lora_dir']}</li>
    </ul>
    <hr>
    <p>Training log:</p>
    <pre>{json.dumps(results['log'], indent=2)}</pre>
    """
    send_email(to_email, subject, body)


def notify_error(to_email: str, error: str):
    """Send training error notification."""
    subject = "❌ AuraBook Training Failed"
    body = f"""
    <h2>Training Failed</h2>
    <p><strong>Error:</strong></p>
    <pre>{error}</pre>
    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    send_email(to_email, subject, body)


# ═══════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r" {2,}", " ", text)

def is_valid(text):
    if not text or len(text) < 10 or len(text) > 8192: return False
    return 3 <= len(text.split()) <= 4096

def format_chat(u, a):
    return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{u}</s>\n<|assistant|>\n{a}</s>"

seen_hashes = set()
def is_dup(text):
    h = hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()
    if h in seen_hashes: return True
    seen_hashes.add(h)
    return False

def fmt_example(raw, fields):
    texts = {t: str(raw.get(s, "") or "") for t, s in fields.items()}
    main = texts.get("output", texts.get("response", texts.get("answer", texts.get("func_code_string", ""))))
    main = clean_text(main)
    if not is_valid(main) or is_dup(main): return None
    user = texts.get("instruction", texts.get("query", texts.get("question", texts.get("func_documentation_string", ""))))
    ctx = texts.get("input", texts.get("context", ""))
    if ctx and ctx.strip(): user = f"{user}\n\nContext:\n{ctx}"
    user = clean_text(user)
    if not user or len(user) < 5: return None
    return format_chat(user, main)

def load_batch(target):
    examples = []
    for cat, cfg in DATASETS.items():
        cat_target = int(target * cfg["ratio"])
        for repo, subset, split, fields in cfg["sources"]:
            try:
                ds = load_dataset(repo, subset, split=split, streaming=True)
                count = 0
                for i, ex in enumerate(ds):
                    if count >= cat_target // len(cfg["sources"]): break
                    f = fmt_example(ex, fields)
                    if f:
                        h = hashlib.md5(f.encode()).hexdigest()
                        if h not in seen_hashes:
                            examples.append(f)
                            seen_hashes.add(h)
                            count += 1
            except Exception as e:
                print(f"  Warning: {repo}: {e}")
    random.shuffle(examples)
    return examples[:target]


# ═══════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════

def train(to_email: str, total_samples: int = 100000, batch_size: int = 10000):
    """Main training function.

    Args:
        to_email: Email to notify.
        total_samples: Total training samples.
        batch_size: Samples per round.
    """
    # Config
    model_name = "google/gemma-2-2b-it"
    hf_token = os.environ.get("HF_TOKEN")
    output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    merged_dir = os.path.join(output_dir, "merged_model")
    lora_dir = os.path.join(output_dir, "lora_adapter")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # GPU Detection
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    BF16 = torch.cuda.is_bf16_supported()

    if vram_gb <= 8:
        BS, GA, BS_MAX = 1, 16, 1024
    elif vram_gb <= 16:
        BS, GA, BS_MAX = 2, 8, 1024
    elif vram_gb <= 24:
        BS, GA, BS_MAX = 4, 4, 2048
    else:
        BS, GA, BS_MAX = 8, 2, 2048

    num_rounds = total_samples // batch_size

    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"Model: {model_name}")
    print(f"Total: {total_samples:,} | Per round: {batch_size:,} | Rounds: {num_rounds}")

    # Notify start
    notify_start(to_email, {
        "model_name": model_name,
        "total_samples": total_samples,
        "num_rounds": num_rounds,
    })

    # Load model
    print("Loading model...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    ))

    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t = sum(p.numel() for p in model.parameters())
    print(f"Model ready: {p:,} trainable / {t:,} total ({p/t*100:.2f}%)")

    # Training loop
    train_log = []
    start_time = time.time()

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"  Round {rnd}/{num_rounds}")
        print(f"{'='*60}")

        # Load data
        print("Loading data batch...")
        batch_examples = load_batch(batch_size)
        print(f"  Loaded: {len(batch_examples):,} examples")

        if len(batch_examples) < 100:
            print("  Too few examples, skipping")
            continue

        ds = Dataset.from_dict({"text": batch_examples})
        split_idx = int(len(ds) * 0.95)
        train_ds = ds.select(range(split_idx))
        eval_ds = ds.select(range(split_idx, len(ds)))

        # Training args
        effective_batch = BS * GA
        steps_per_epoch = max(len(train_ds) // effective_batch, 1)

        round_output = os.path.join(output_dir, f"round-{rnd}")
        os.makedirs(round_output, exist_ok=True)

        training_args = SFTConfig(
            output_dir=round_output,
            num_train_epochs=2,
            per_device_train_batch_size=BS,
            gradient_accumulation_steps=GA,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=not BF16, bf16=BF16, tf32=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=min(100, steps_per_epoch),
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to="none",
            run_name=f"aura-round-{rnd}",
            seed=42 + rnd,
            dataset_text_field="text",
            packing=False,
        )

        trainer = SFTTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer, max_seq_length=BS_MAX,
        )

        # Train
        try:
            result = trainer.train()
            loss = result.metrics.get("train_loss", 0)
            runtime = time.time() - t0
            print(f"Round {rnd} done! Loss: {loss:.4f} | Time: {runtime:.1f}s")
            train_log.append({"round": rnd, "loss": loss, "runtime": runtime})
        except Exception as e:
            print(f"Error in round {rnd}: {e}")
            continue

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{rnd}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

        # Free memory
        del batch_examples, ds, train_ds, eval_ds, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Merge model
    print("\nMerging LoRA weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", token=hf_token,
    )
    last_ckpt = os.path.join(checkpoint_dir, f"checkpoint-{num_rounds}")
    merged_model = PeftModel.from_pretrained(base_model, last_ckpt)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    os.makedirs(lora_dir, exist_ok=True)
    model.save_pretrained(lora_dir)

    print(f"Merged model saved: {merged_dir}")

    # Results
    total_time = time.time() - start_time
    results = {
        "total_rounds": num_rounds,
        "final_loss": train_log[-1]["loss"] if train_log else 0,
        "total_time": total_time,
        "merged_dir": merged_dir,
        "lora_dir": lora_dir,
        "log": train_log,
    }

    # Save results
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Notify complete
    notify_complete(to_email, results)

    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraBook Training Service")
    parser.add_argument("--email", required=True, help="Email for notifications")
    parser.add_argument("--size", default="100k", help="Dataset size")
    parser.add_argument("--batch", type=int, default=10000, help="Batch size per round")
    args = parser.parse_args()

    size_map = {"50k": 50000, "100k": 100000, "250k": 250000, "500k": 500000, "1M": 1000000}
    total = size_map.get(args.size, 100000)

    try:
        train(args.email, total, args.batch)
    except Exception as e:
        notify_error(args.email, str(e))
        raise
