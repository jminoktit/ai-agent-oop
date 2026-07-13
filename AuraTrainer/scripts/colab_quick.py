#!/usr/bin/env python3
"""Quick Colab Training - Minimal Version.

Paste each section between # === markers as a separate Colab cell.
Runtime → Change runtime type → T4 GPU
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Install & Setup                                ║
# ╚══════════════════════════════════════════════════════════╝

!pip install -q torch transformers datasets peft bitsandbytes trl accelerate pyyaml safetensors huggingface-hub

from google.colab import drive
drive.mount("/content/drive")

import os, torch
os.makedirs("/content/drive/MyDrive/AuraBook/outputs", exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Auto-Configure                                 ║
# ╚══════════════════════════════════════════════════════════╝

gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

if vram <= 8:
    BS, GA, BS_MAX = 1, 16, 1024
elif vram <= 16:
    BS, GA, BS_MAX = 2, 8, 1024
elif vram <= 24:
    BS, GA, BS_MAX = 4, 4, 2048
elif vram <= 40:
    BS, GA, BS_MAX = 8, 2, 2048
else:
    BS, GA, BS_MAX = 16, 1, 4096

BF16 = torch.cuda.is_bf16_supported()
print(f"Batch: {BS}, GradAccum: {GA}, Effective: {BS*GA}, MaxLen: {BS_MAX}, BF16: {BF16}")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: Load Data (Streaming)                          ║
# ╚══════════════════════════════════════════════════════════╝

from datasets import load_dataset, Dataset
import hashlib, random, re

SYSTEM = "You are Aura, a helpful university assistant. You explain programming, write code, debug errors, explain math, and help study. You speak Arabic and English."

def clean(t):
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(t).strip())
    return re.sub(r"\n{3,}", "\n\n", re.sub(r" {2,}", " ", t))

seen = set()
def ok(t):
    t = clean(t)
    if len(t) < 10 or len(t) > 8192: return False
    h = hashlib.md5(" ".join(t.lower().split()).encode()).hexdigest()
    if h in seen: return False
    seen.add(h)
    return len(t.split()) >= 3

def fmt(u, a):
    return f"<|system|>\n{SYSTEM}</s>\n<|user|>\n{u}</s>\n<|assistant|>\n{a}</s>"

TARGET = 100000
all_data = []

# Programming 40%
for repo, sub, flds in [
    ("tatsu-lab/alpaca", None, ("instruction","input","output")),
    ("google-research-datasets/mbpp", "full", ("problem","solution","problem")),
]:
    ds = load_dataset(repo, sub, split="train", streaming=True)
    n = 0
    for ex in ds:
        if n >= 20000: break
        texts = [str(ex.get(f,"") or "") for f in flds]
        if ok(texts[2]):
            all_data.append(fmt(f"{texts[0]}\n{texts[1]}" if texts[1] else texts[0], clean(texts[2])))
            n += 1
    print(f"  {repo}: {n}")

# Chat 25%
for repo, sub, flds in [
    ("Open-Orca/OpenOrca-Platypus2", None, ("instruction","input","output")),
    ("stingning/ultrachat", None, ("prompt","prompt","response")),
]:
    ds = load_dataset(repo, sub, split="train", streaming=True)
    n = 0
    for ex in ds:
        if n >= 15000: break
        texts = [str(ex.get(f,"") or "") for f in flds]
        if ok(texts[2]):
            all_data.append(fmt(texts[0], clean(texts[2])))
            n += 1
    print(f"  {repo}: {n}")

# Math 15%
for repo, sub, flds in [
    ("meta-math/MetaMathQA", None, ("query","response","response")),
    ("openai/gsm8k", "main", ("question","answer","answer")),
]:
    ds = load_dataset(repo, sub, split="train", streaming=True)
    n = 0
    for ex in ds:
        if n >= 10000: break
        texts = [str(ex.get(f,"") or "") for f in flds]
        if ok(texts[2]):
            all_data.append(fmt(f"Solve step by step:\n{texts[0]}", clean(texts[2])))
            n += 1
    print(f"  {repo}: {n}")

# Arabic 10%
for repo, sub, flds in [
    ("MBZUAI-AILab/arabic-openhermes", None, ("instruction","input","output")),
]:
    ds = load_dataset(repo, sub, split="train", streaming=True)
    n = 0
    for ex in ds:
        if n >= 5000: break
        texts = [str(ex.get(f,"") or "") for f in flds]
        if ok(texts[2]):
            all_data.append(fmt(texts[0], clean(texts[2])))
            n += 1
    print(f"  {repo}: {n}")

random.shuffle(all_data)
all_data = all_data[:TARGET]

split = int(len(all_data) * 0.95)
train_ds = Dataset.from_dict({"text": all_data[:split]})
eval_ds = Dataset.from_dict({"text": all_data[split:]})
print(f"\n✅ Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Load Model + QLoRA                             ║
# ╚══════════════════════════════════════════════════════════╝

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb, device_map="auto", trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
))

p = sum(p.numel() for p in model.parameters() if p.requires_grad)
t = sum(p.numel() for p in model.parameters())
print(f"✅ Model ready: {p:,} trainable / {t:,} total ({p/t*100:.2f}%)")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Train                                          ║
# ╚══════════════════════════════════════════════════════════╝

OUTPUT = "/content/drive/MyDrive/AuraBook/outputs"

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUTPUT,
        num_train_epochs=3,
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
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        run_name="aura-book-qlora",
        seed=42,
        max_seq_length=BS_MAX,
        dataset_text_field="text",
        packing=False,
    ),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,
)

# Resume if checkpoint exists
import glob
ckpts = sorted(glob.glob(f"{OUTPUT}/checkpoint-*"))
resume = ckpts[-1] if ckpts else None
if resume: print(f"📂 Resuming: {resume}")

print("🚀 Training started...")
result = trainer.train(resume_from_checkpoint=resume)
print(f"✅ Done! Loss: {result.metrics['train_loss']:.4f}")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6: Save & Test                                    ║
# ╚══════════════════════════════════════════════════════════╝

# Save LoRA
model.save_pretrained(f"{OUTPUT}/lora_adapter")
tok.save_pretrained(f"{OUTPUT}/lora_adapter")

# Merge & save
merged = model.merge_and_unload()
merged.save_pretrained(f"{OUTPUT}/merged", safe_serialization=True)
tok.save_pretrained(f"{OUTPUT}/merged")

# Test
from transformers import pipeline
pipe = pipeline("text-generation", model=f"{OUTPUT}/merged", tokenizer=tok, torch_dtype=torch.float16, device_map="auto")

def ask(q):
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":q}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    r = pipe(t, max_new_tokens=512, temperature=0.7, do_sample=True)
    resp = r[0]["generated_text"]
    return resp.split("<|assistant|>")[-1].replace("</s>","").strip() if "<|assistant|>" in resp else resp

for q in [
    "Write a Python function to find factorial",
    "What is Big O notation?",
    "اشرح لي الفرق بين Class و Object",
]:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")

print(f"\n✅ Saved to: {OUTPUT}")
