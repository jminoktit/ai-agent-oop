# AuraBook - TinyLlama QLoRA Fine-tuning

AI University Assistant trained with QLoRA on TinyLlama-1.1B-Chat.

## Quick Start (Google Colab)

### Option 1: Single Cell (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Create new notebook
4. Paste this in ONE cell and run:

```python
!git clone https://github.com/jminoktit/ai-agent-oop.git /content/ai-agent-oop
!python /content/ai-agent-oop/AuraTrainer/scripts/colab_onecell.py
```

### Option 2: Multiple Cells
```python
# Cell 1
!git clone https://github.com/jminoktit/ai-agent-oop.git
%cd ai-agent-oop/AuraTrainer
!python scripts/colab_sequential.py
```

## Local Training

```bash
# Setup
python setup_local.py

# Train (100K samples)
python -m AuraTrainer.scripts.train --size 100k

# Test
python -m AuraTrainer.scripts.inference --model-path ./outputs/merged_model --interactive
```

## Features

- QLoRA 4-bit NF4 quantization
- Auto GPU detection (T4, L4, V100, A100, RTX 3060/4060/4090)
- Sequential batch training (10K per round) for Colab
- Checkpoint saving to Google Drive
- Auto-resume if interrupted
- Model merging (LoRA + base)
- Supports: Python, SQL, Math, Arabic, English

## Dataset Sizes

| Size | Time (T4 GPU) | Time (RTX 4090) |
|------|--------------|-----------------|
| 50K  | ~30 min      | ~10 min         |
| 100K | ~1 hour      | ~20 min         |
| 250K | ~2.5 hours   | ~45 min         |
| 500K | ~5 hours     | ~1.5 hours      |
| 1M   | ~10 hours    | ~3 hours        |
