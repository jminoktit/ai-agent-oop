#!/usr/bin/env python3
"""
AuraBook Colab Training - FINAL ROBUST VERSION
Copy entire contents into ONE Colab cell and run.
GPU: Runtime -> Change runtime type -> T4 GPU
"""
import os, subprocess, sys

# CRITICAL: Avoid local module conflicts
os.chdir('/content')
sys.path = [p for p in sys.path if 'AuraTrainer' not in p and 'ai-agent-op' not in p]

# === STEP 1: Clone & Install ===
print("Setting up environment...")
subprocess.run(['rm', '-rf', '/content/ai-agent-oop'], check=False)

print("Cloning repo...")
r = subprocess.run(
    ['git', 'clone', 'https://github.com/jminoktit/ai-agent-oop.git', '/content/ai-agent-oop'],
    capture_output=True, text=True
)
if r.returncode != 0:
    print(f"Clone failed: {r.stderr}")
    sys.exit(1)
print("Repo cloned!")

print("Installing packages...")
for p in ["torch>=2.1.0", "transformers>=4.36.0", "datasets>=2.16.0",
          "peft>=0.7.0", "bitsandbytes>=0.41.0", "trl>=0.7.4",
          "accelerate>=0.25.0", "safetensors>=0.4.0", "huggingface-hub>=0.19.0"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
print("Packages installed!")

# Mount Google Drive (optional)
GDRIVE = False
try:
    from google.colab import drive
    drive.mount("/content/drive")
    GDRIVE = True
    print("Google Drive mounted!")
except Exception:
    print("Google Drive not available, using local storage")

# === STEP 2: Imports & Config ===
import torch, gc, json, time, hashlib, random, re, glob

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
BF16 = torch.cuda.is_bf16_supported()

if vram_gb <= 8:
    BS, GA, BS_MAX = 1, 16, 1024
elif vram_gb <= 16:
    BS, GA, BS_MAX = 2, 8, 1024
elif vram_gb <= 24:
    BS, GA, BS_MAX = 4, 4, 2048
elif vram_gb <= 40:
    BS, GA, BS_MAX = 8, 2, 2048
else:
    BS, GA, BS_MAX = 16, 1, 4096

if GDRIVE:
    CKPT_DIR = "/content/drive/MyDrive/AuraBook/checkpoints"
    OUT_DIR = "/content/drive/MyDrive/AuraBook/outputs"
else:
    CKPT_DIR = "/content/ai-agent-oop/AuraTrainer/outputs/checkpoints"
    OUT_DIR = "/content/ai-agent-oop/AuraTrainer/outputs"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG = {
    "TOTAL_SAMPLES": 100000,
    "BATCH_SIZE": 10000,
    "EVAL_SPLIT": 0.05,
    "MODEL_NAME": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "LORA_R": 64, "LORA_ALPHA": 128, "LORA_DROPOUT": 0.05,
    "EPOCHS_PER_ROUND": 2,
    "LEARNING_RATE": 2e-4,
    "WARMUP_RATIO": 0.03,
    "MAX_SEQ_LENGTH": min(2048, BS_MAX),
    "SAVE_EVERY_N_STEPS": 100,
    "BATCH_SIZE_GPU": BS, "GRAD_ACCUM": GA,
    "CHECKPOINT_DIR": CKPT_DIR,
    "OUTPUT_DIR": OUT_DIR,
}

print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
print(f"Batch: {BS}, GradAccum: {GA}, Effective: {BS*GA}")
print(f"BF16: {BF16}, MaxLen: {CONFIG['MAX_SEQ_LENGTH']}")
print(f"Total: {CONFIG['TOTAL_SAMPLES']:,} samples in {CONFIG['TOTAL_SAMPLES']//CONFIG['BATCH_SIZE']} rounds")

# === STEP 3: Format Functions ===
SYSTEM_PROMPT = (
    "You are Aura, a helpful university assistant specializing in programming, "
    "mathematics, and education. You can explain programming concepts, write code, "
    "debug errors, explain math problems, and help with studying. You speak both "
    "Arabic and English fluently."
)

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

print("Format functions ready")

# === STEP 4: Load Data (Streaming) ===
from datasets import load_dataset as hf_load_dataset, Dataset

# ALL VERIFIED WORKING DATASETS
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
                ds = hf_load_dataset(repo, subset, split=split, streaming=True)
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

print("Data generator ready")

# === STEP 5: Load Model + QLoRA ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(f"Loading model: {CONFIG['MODEL_NAME']}")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["MODEL_NAME"], quantization_config=bnb, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=CONFIG["LORA_R"], lora_alpha=CONFIG["LORA_ALPHA"],
    lora_dropout=CONFIG["LORA_DROPOUT"], bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
))

p = sum(p.numel() for p in model.parameters() if p.requires_grad)
t = sum(p.numel() for p in model.parameters())
print(f"Model ready: {p:,} trainable / {t:,} total ({p/t*100:.2f}%)")

# === STEP 6: Training Loop ===
from trl import SFTTrainer, SFTConfig

TOTAL = CONFIG["TOTAL_SAMPLES"]
BATCH = CONFIG["BATCH_SIZE"]
NUM_ROUNDS = TOTAL // BATCH

print("=" * 60)
print(f"  Starting Sequential Training")
print(f"  Total: {TOTAL:,} | Per round: {BATCH:,} | Rounds: {NUM_ROUNDS}")
print("=" * 60)

resume_ckpt = None
for ckpt in sorted(glob.glob(f"{CONFIG['CHECKPOINT_DIR']}/checkpoint-*")):
    try:
        r = int(os.path.basename(ckpt).split("-")[1])
        resume_ckpt = ckpt
    except: pass
if resume_ckpt:
    print(f"Resuming from: {resume_ckpt}")

train_log = []

for rnd in range(1, NUM_ROUNDS + 1):
    t0 = time.time()

    if resume_ckpt:
        try:
            done = int(os.path.basename(resume_ckpt).split("-")[1])
            if rnd <= done:
                print(f"Round {rnd} done, skipping")
                continue
        except: pass

    print(f"\n{'='*60}")
    print(f"  Round {rnd}/{NUM_ROUNDS}")
    print(f"{'='*60}")

    print("Loading data batch...")
    batch_examples = load_batch(BATCH)
    print(f"  Loaded: {len(batch_examples):,} examples")

    if len(batch_examples) < 100:
        print("  Too few examples, skipping")
        continue

    ds = Dataset.from_dict({"text": batch_examples})
    split_idx = int(len(ds) * (1 - CONFIG["EVAL_SPLIT"]))
    train_ds = ds.select(range(split_idx))
    eval_ds = ds.select(range(split_idx, len(ds)))
    print(f"  Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    effective_batch = CONFIG["BATCH_SIZE_GPU"] * CONFIG["GRAD_ACCUM"]
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * CONFIG["EPOCHS_PER_ROUND"]
    print(f"  Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    if steps_per_epoch == 0:
        print("  Warning: batch too large for dataset, adjusting...")
        CONFIG["GRAD_ACCUM"] = 1
        effective_batch = CONFIG["BATCH_SIZE_GPU"]
        steps_per_epoch = len(train_ds) // effective_batch
        total_steps = steps_per_epoch * CONFIG["EPOCHS_PER_ROUND"]

    round_output = f"{CONFIG['OUTPUT_DIR']}/round-{rnd}"
    os.makedirs(round_output, exist_ok=True)

    training_args = SFTConfig(
        output_dir=round_output,
        num_train_epochs=CONFIG["EPOCHS_PER_ROUND"],
        per_device_train_batch_size=CONFIG["BATCH_SIZE_GPU"],
        gradient_accumulation_steps=CONFIG["GRAD_ACCUM"],
        learning_rate=CONFIG["LEARNING_RATE"],
        lr_scheduler_type="cosine",
        warmup_ratio=CONFIG["WARMUP_RATIO"],
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=not BF16, bf16=BF16, tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=CONFIG["SAVE_EVERY_N_STEPS"],
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=min(CONFIG["SAVE_EVERY_N_STEPS"], max(steps_per_epoch, 50)),
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
        processing_class=tokenizer,
        max_seq_length=CONFIG["MAX_SEQ_LENGTH"],
    )

    print("Training...")
    try:
        result = trainer.train()
        loss = result.metrics.get("train_loss", 0)
        runtime = time.time() - t0
        print(f"Round {rnd} done! Loss: {loss:.4f} | Time: {runtime:.1f}s")
        train_log.append({"round": rnd, "loss": loss, "runtime": runtime, "samples": len(train_ds)})
    except torch.cuda.OutOfMemoryError:
        print(f"OOM! Reducing batch size...")
        torch.cuda.empty_cache()
        CONFIG["BATCH_SIZE_GPU"] = max(1, CONFIG["BATCH_SIZE_GPU"] // 2)
        CONFIG["GRAD_ACCUM"] = min(CONFIG["GRAD_ACCUM"] * 2, 32)
        print(f"  New: batch={CONFIG['BATCH_SIZE_GPU']}, grad_accum={CONFIG['GRAD_ACCUM']}")
        continue
    except Exception as e:
        print(f"Error in round {rnd}: {e}")
        continue

    ckpt_path = f"{CONFIG['CHECKPOINT_DIR']}/checkpoint-{rnd}"
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    with open(f"{CONFIG['CHECKPOINT_DIR']}/training_state.json", "w") as f:
        json.dump({"round": rnd, "total_rounds": NUM_ROUNDS, "log": train_log}, f, indent=2)

    del batch_examples, ds, train_ds, eval_ds, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared")

    elapsed = time.time() - t0
    eta = (NUM_ROUNDS - rnd) * elapsed
    print(f"Progress: {rnd}/{NUM_ROUNDS} ({rnd/NUM_ROUNDS*100:.0f}%) | ETA: {eta/60:.1f} min")

print(f"\nAll {NUM_ROUNDS} rounds completed!")

# === STEP 7: Merge Model ===
from peft import PeftModel

print("Merging LoRA weights...")
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["MODEL_NAME"], torch_dtype=torch.float16, device_map="auto",
)
last_ckpt = f"{CONFIG['CHECKPOINT_DIR']}/checkpoint-{NUM_ROUNDS}"
merged_model = PeftModel.from_pretrained(base_model, last_ckpt)
merged_model = merged_model.merge_and_unload()

merged_dir = f"{CONFIG['OUTPUT_DIR']}/merged_model"
os.makedirs(merged_dir, exist_ok=True)
merged_model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print(f"Merged model saved: {merged_dir}")

del base_model, merged_model
gc.collect()
torch.cuda.empty_cache()

# === STEP 8: Test ===
from transformers import pipeline

print("Loading merged model for testing...")
pipe = pipeline("text-generation", model=merged_dir, tokenizer=tokenizer,
                torch_dtype=torch.float16, device_map="auto")

def ask_aura(prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(formatted, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    resp = output[0]["generated_text"]
    if "<|assistant|>" in resp: resp = resp.split("<|assistant|>")[-1]
    return resp.replace("</s>", "").strip()

for cat, q in [
    ("Python", "Write a Python function to check if a number is prime."),
    ("SQL", "Write a SQL query to find duplicate emails in a Users table."),
    ("Math", "Explain the Pythagorean theorem."),
    ("Arabic", "What is Big O notation? Explain in Arabic."),
    ("English", "What is the time complexity of binary search?"),
]:
    print(f"\n{cat}: {q}")
    print(f"Aura: {ask_aura(q)}")

print("\nTraining Summary:")
for e in train_log:
    print(f"  Round {e['round']}: Loss={e['loss']:.4f} | Time={e['runtime']:.1f}s")
print(f"\nModels saved: {CONFIG['OUTPUT_DIR']}")
print("DONE!")
