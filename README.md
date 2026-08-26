# Mistral-7B LLM Custom Training

This project explores custom fine-tuning of the Mistral-7B language model on consumer hardware, with a focus on reducing computational and memory requirements while maintaining high-quality text generation.

## Overview

The pipeline fine tunes Mistral-7B-Instruct on a custom conversational dataset using LoRA (parameter efficient fine tuning) on top of a 4-bit quantized base model, then batched inference with the resulting adapter.

```
Base Mistral-7B Model
        │
        ▼
4-bit Quantization (NF4)
        │
        ▼
LoRA Fine-Tuning
        │
        ▼
Optimized Batched Inference
        │
        ▼
Local Deployment
```

## Features

- LoRA fine-tuning pipeline
- 4-bit quantization using BitsAndBytes
- Batched inference support
- Optimized VRAM usage
- Local deployment support
- Multi-prompt generation
- Efficient tokenizer handling
- GPU acceleration with CUDA

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM) |
| CPU | Intel i5-11400/F (2.60 GHz base, 4.40 GHz boost) |
| RAM | 16GB (32GB recommended) |
| Storage | 500GB SSD + 1TB HDD (full 1TB not required — the model and adapters take up less space) |

This project was built to demonstrate that LLM fine-tuning is achievable on consumer hardware and not just data center GPUs.

## Data Collection

Training data was sourced from YouTube transcripts and formatted into 423 total training prompts. Each example follows a user/assistant chat schema:

```json
{"messages": [{"role": "user", "content": "message here"}, {"role": "assistant", "content": "message here"}]}
```

Data is in the form `train.jsonl`, one JSON object per line.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- BitsAndBytes
- CUDA
- Mistral-7B

## Optimizations Implemented

**Quantization** — 4-bit NF4 quantization (via BitsAndBytes) cuts VRAM usage to load a 7B model on a 12GB GPU.

**LoRA Fine-Tuning** — Parameter fine tuning trains a small set of adapter weights instead of the full model, reducing training cost.

**Batched Inference** — Multi prompt batching improves GPU utilization during generation (allows for multiple prompts for the same GPU usage).

**CUDA Optimization** — GPU acceleration and tensor handling are tuned for faster performance on limited VRAM.

## Project Structure

```
.
├── train.py          # LoRA fine-tuning script
├── run.py            # Batched inference script
├── train.jsonl        # Training data (user/assistant message pairs)
├── requirements.txt   # Python dependencies
├── lora_out/          # Saved LoRA adapter + tokenizer (generated after training)
└── README.md
```

## Implementation tutorial

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

A CUDA capable GPU with at least ~12GB VRAM is recommended as that is what I used.

### 2. Prepare training data

Add your training examples to `train.jsonl`, following the message schema described above.

### 3. Fine-tune

```bash
python train.py
```

This loads the base model in 4-bit, applies a LoRA adapter, trains on `train.jsonl`, and saves the resulting adapter to `lora_out/`.

### 4. Run inference

```bash
python run.py
```

This loads the base model plus the trained LoRA adapter and generates responses for a batch of example prompts.

## Notes

- Both scripts currently point at `mistralai/Mistral-7B-Instruct-v0.2` as the base model — swap `BASE_MODEL` / `MODEL_ID` if you want to fine-tune a different checkpoint.
- Training config (batch size, epochs, learning rate, LoRA rank, etc.) is defined inline in `train.py`. Adjust as needed.


