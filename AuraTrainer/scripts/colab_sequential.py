#!/usr/bin/env python3
"""Google Colab Training - Sequential Batch Approach (FIXED).

Smart training that loads 10k samples at a time, trains, saves,
and frees memory before loading the next batch.

Usage in Colab:
    Run each cell between # === markers as a separate cell.
    Runtime → Change runtime type → T4 GPU
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Install & Setup                                ║
# ╚══════════════════════════════════════════════════════════╝

import subprocess, sys, os

# CRITICAL: Change to /content to avoid local module import conflicts
os.chdir('/content')
sys.path = [p for p in sys.path if 'AuraTrainer' not in p and 'ai-agent-op' not in p]

def install():
    pkgs = [
        "torch>=2.1.0", "transformers>=4.36.0", "datasets>=2.16.0",
        "peft>=0.7.0", "bitsandbytes>=0.41.0", "trl>=0.7.4",
        "accelerate>=0.25.0", "pyyaml>=6.0", "tqdm>=4.65.0",
        "safetensors>=0.4.0", "huggingface-hub>=0.19.0",
    ]
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
    print("✅ Packages installed")

install()

from google.colab import drive
drive.mount("/content/drive")

import os, torch, gc, json, time, hashlib, random, re, glob
from pathlib import Path

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Configuration                                  ║
# ╚══════════════════════════════════════════════════════════╝

CONFIG = {
    "TOTAL_SAMPLES": 100000,
    "BATCH_SIZE": 10000,
    "EVAL_SPLIT": 0.05,
    "MODEL_NAME": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "LORA_R": 64,
    "LORA_ALPHA": 128,
    "LORA_DROPOUT": 0.05,
    "EPOCHS_PER_ROUND": 2,
    "LEARNING_RATE": 2e-4,
    "WARMUP_RATIO": 0.03,
    "MAX_SEQ_LENGTH": 2048,
    "SAVE_EVERY_N_STEPS": 100,
    "KEEP_LAST_N_CHECKPOINTS": 3,
    "GDRIVE_BASE": "/content/drive/MyDrive/AuraBook",
    "CHECKPOINT_DIR": "/content/drive/MyDrive/AuraBook/checkpoints",
    "OUTPUT_DIR": "/content/drive/MyDrive/AuraBook/outputs",
}

os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

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

BF16 = torch.cuda.is_bf16_supported()
CONFIG["BATCH_SIZE_GPU"] = BS
CONFIG["GRAD_ACCUM"] = GA
CONFIG["MAX_SEQ_LENGTH"] = min(CONFIG["MAX_SEQ_LENGTH"], BS_MAX)

print(f"🖥️  GPU: {gpu_name} ({vram_gb:.1f} GB)")
print(f"📦 Batch: {BS}, GradAccum: {GA}, Effective: {BS*GA}")
print(f"⚡ BF16: {BF16}, MaxLen: {CONFIG['MAX_SEQ_LENGTH']}")
print(f"🎯 Total: {CONFIG['TOTAL_SAMPLES']:,} samples in {CONFIG['TOTAL_SAMPLES']//CONFIG['BATCH_SIZE']} rounds")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: System Prompt & Format Functions                ║
# ╚══════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = (
    "You are Aura, a helpful university assistant specializing in programming, "
    "mathematics, and education. You can explain programming concepts, write code, "
    "debug errors, explain math problems, and help with studying. You speak both "
    "Arabic and English fluently."
)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text

def is_valid(text):
    if not text or len(text) < 10 or len(text) > 8192:
        return False
    words = text.split()
    return 3 <= len(words) <= 4096

def format_chat(user_msg, assistant_msg):
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n{assistant_msg}</s>"
    )

seen_hashes = set()
def is_duplicate(text):
    h = hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False

print("✅ Format functions ready")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Data Generator (Streaming)                      ║
# ╚══════════════════════════════════════════════════════════╝

from datasets import load_dataset, Dataset

# ─── VERIFIED WORKING DATASETS (No trust_remote_code) ────
DATASETS = {
    "programming": {
        "ratio": 0.40,
        "sources": [
            ("tatsu-lab/alpaca", None, "train",
             {"instruction": "instruction", "input": "input", "output": "output"}),
            ("BAAI/CodeSearchNet", "python", "train",
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
            ("OALL/Arabic-Alpaca-20K", None, "train",
             {"instruction": "instruction", "input": "input", "output": "output"}),
        ],
    },
}

def format_example(raw, fields):
    """Format a raw example based on fields mapping."""
    texts = {}
    for target, source in fields.items():
        texts[target] = str(raw.get(source, "") or "")

    main_text = texts.get("output", texts.get("response", texts.get("answer", texts.get("func_code_string", texts.get("code", "")))))
    main_text = clean_text(main_text)

    if not is_valid(main_text) or is_duplicate(main_text):
        return None

    user = texts.get("instruction", texts.get("query", texts.get("question", texts.get("func_documentation_string", ""))))
    context = texts.get("input", texts.get("context", ""))

    if context and context.strip():
        user = f"{user}\n\nContext:\n{context}"

    user = clean_text(user)
    if not user or len(user) < 5:
        return None

    return format_chat(user, main_text)


def load_batch_streaming(target_count, batch_idx=0, seen_hashes_local=None):
    """Load a batch of examples using streaming."""
    if seen_hashes_local is None:
        seen_hashes_local = set()

    examples = []

    for cat_name, cat_config in DATASETS.items():
        ratio = cat_config["ratio"]
        cat_target = int(target_count * ratio)

        for repo, subset, split, fields in cat_config["sources"]:
            try:
                ds = load_dataset(repo, subset, split=split, streaming=True)

                count = 0
                for i, ex in enumerate(ds):
                    if count >= cat_target // len(cat_config["sources"]):
                        break

                    formatted = format_example(ex, fields)
                    if formatted:
                        h = hashlib.md5(formatted.encode()).hexdigest()
                        if h not in seen_hashes_local:
                            examples.append(formatted)
                            seen_hashes_local.add(h)
                            count += 1

            except Exception as e:
                print(f"  ⚠️  {repo}: {e}")
                continue

    random.shuffle(examples)
    return examples[:target_count]

print("✅ Data generator ready")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Load Model & Apply QLoRA                        ║
# ╚══════════════════════════════════════════════════════════╝

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(f"📥 Loading model: {CONFIG['MODEL_NAME']}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["MODEL_NAME"],
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=CONFIG["LORA_R"],
    lora_alpha=CONFIG["LORA_ALPHA"],
    lora_dropout=CONFIG["LORA_DROPOUT"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Model ready: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6: Training Loop (Sequential Batches)              ║
# ╚══════════════════════════════════════════════════════════╝

from trl import SFTTrainer, SFTConfig

TOTAL = CONFIG["TOTAL_SAMPLES"]
BATCH = CONFIG["BATCH_SIZE"]
NUM_ROUNDS = TOTAL // BATCH
SEEN_HASHES = set()

print("=" * 60)
print(f"  🚀 Starting Sequential Training")
print(f"  Total: {TOTAL:,} samples | Per round: {BATCH:,} | Rounds: {NUM_ROUNDS}")
print("=" * 60)

resume_ckpt = None
existing_ckpts = sorted(glob.glob(f"{CONFIG['CHECKPOINT_DIR']}/checkpoint-*"))
if existing_ckpts:
    resume_ckpt = existing_ckpts[-1]
    ckpt_name = os.path.basename(resume_ckpt)
    try:
        last_round = int(ckpt_name.split("-")[1])
        print(f"📂 Resuming from round {last_round}: {resume_ckpt}")
    except ValueError:
        resume_ckpt = None

training_log = []

for round_num in range(1, NUM_ROUNDS + 1):
    round_start = time.time()

    if resume_ckpt:
        try:
            completed_round = int(os.path.basename(resume_ckpt).split("-")[1])
            if round_num <= completed_round:
                print(f"\n⏭️  Skipping round {round_num} (already completed)")
                continue
        except ValueError:
            pass

    print(f"\n{'='*60}")
    print(f"  📦 Round {round_num}/{NUM_ROUNDS}")
    print(f"{'='*60}")

    print("📥 Loading data batch...")
    batch_examples = load_batch_streaming(
        BATCH,
        batch_idx=round_num,
        seen_hashes_local=SEEN_HASHES,
    )
    print(f"   Loaded: {len(batch_examples):,} examples")

    if len(batch_examples) < 100:
        print("   ⚠️  Too few examples, skipping round")
        continue

    ds = Dataset.from_dict({"text": batch_examples})

    split_idx = int(len(ds) * (1 - CONFIG["EVAL_SPLIT"]))
    train_ds = ds.select(range(split_idx))
    eval_ds = ds.select(range(split_idx, len(ds)))

    print(f"   Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    effective_batch = CONFIG["BATCH_SIZE_GPU"] * CONFIG["GRAD_ACCUM"]
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * CONFIG["EPOCHS_PER_ROUND"]

    print(f"   Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    round_output = f"{CONFIG['OUTPUT_DIR']}/round-{round_num}"
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
        fp16=not BF16,
        bf16=BF16,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=CONFIG["SAVE_EVERY_N_STEPS"],
        save_total_limit=CONFIG["KEEP_LAST_N_CHECKPOINTS"],
        eval_strategy="steps",
        eval_steps=min(CONFIG["SAVE_EVERY_N_STEPS"], max(steps_per_epoch, 50)),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        run_name=f"aura-round-{round_num}",
        seed=42 + round_num,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        max_seq_length=CONFIG["MAX_SEQ_LENGTH"],
    )

    print("🚀 Training...")
    try:
        result = trainer.train()
        loss = result.metrics.get("train_loss", 0)
        runtime = time.time() - round_start
        print(f"✅ Round {round_num} done! Loss: {loss:.4f} | Time: {runtime:.1f}s")
        training_log.append({
            "round": round_num,
            "loss": loss,
            "runtime": runtime,
            "samples": len(train_ds),
        })
    except torch.cuda.OutOfMemoryError:
        print(f"⚠️  OOM! Reducing batch size...")
        torch.cuda.empty_cache()
        CONFIG["BATCH_SIZE_GPU"] = max(1, CONFIG["BATCH_SIZE_GPU"] // 2)
        CONFIG["GRAD_ACCUM"] = min(CONFIG["GRAD_ACCUM"] * 2, 32)
        print(f"   New: batch={CONFIG['BATCH_SIZE_GPU']}, grad_accum={CONFIG['GRAD_ACCUM']}")
        continue

    ckpt_path = f"{CONFIG['CHECKPOINT_DIR']}/checkpoint-{round_num}"
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")

    state = {
        "round": round_num,
        "total_rounds": NUM_ROUNDS,
        "total_samples": TOTAL,
        "samples_trained": round_num * BATCH,
        "log": training_log,
    }
    with open(f"{CONFIG['CHECKPOINT_DIR']}/training_state.json", "w") as f:
        json.dump(state, f, indent=2)

    del batch_examples, ds, train_ds, eval_ds
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🧹 Memory cleared")

    elapsed = time.time() - round_start
    remaining = NUM_ROUNDS - round_num
    eta = remaining * elapsed
    print(f"📊 Progress: {round_num}/{NUM_ROUNDS} ({round_num/NUM_ROUNDS*100:.0f}%)")
    print(f"   ETA: {eta/60:.1f} min")

print(f"\n{'='*60}")
print(f"  🎉 All {NUM_ROUNDS} rounds completed!")
print(f"{'='*60}")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7: Merge Final Model                               ║
# ╚══════════════════════════════════════════════════════════╝

from peft import PeftModel

print("🔄 Merging LoRA weights into base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["MODEL_NAME"],
    torch_dtype=torch.float16,
    device_map="auto",
)

last_ckpt = f"{CONFIG['CHECKPOINT_DIR']}/checkpoint-{NUM_ROUNDS}"
merged_model = PeftModel.from_pretrained(base_model, last_ckpt)
merged_model = merged_model.merge_and_unload()

merged_dir = f"{CONFIG['OUTPUT_DIR']}/merged_model"
os.makedirs(merged_dir, exist_ok=True)
merged_model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)

print(f"✅ Merged model saved: {merged_dir}")

del base_model, merged_model
gc.collect()
torch.cuda.empty_cache()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8: Test the Model                                  ║
# ╚══════════════════════════════════════════════════════════╝

from transformers import pipeline

print("📥 Loading merged model for testing...")
pipe = pipeline(
    "text-generation",
    model=merged_dir,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

def ask_aura(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(formatted, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    response = output[0]["generated_text"]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    return response.replace("</s>", "").strip()

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

print("\nTraining Summary:")
for entry in training_log:
    print(f"  Round {entry['round']}: Loss={entry['loss']:.4f} | Time={entry['runtime']:.1f}s")

print(f"\n📁 Models saved: {CONFIG['OUTPUT_DIR']}")
print("🎉 Done!")
