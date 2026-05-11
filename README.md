# Mistral-7B-LLM-custom-trained

This project is focused on using the Mistral-7B LLM model for custom training. Explores methods for reducing computational requirements while maintaining high-quality text generation performance.

## Hardware used 
- NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM)
- Intel i5-11400/F (2.60 GHz base, 4.40 GHz boost)
- 16GB RAM (32GB recommended)
- 500 GB SSD
- 1 TB HDD
(It should be noted that you do not need that much space for the model)

## Features
- LoRA fine-tuning pipeline
- 4-bit quantization using BitsAndBytes
- Batched inference support
- Optimized VRAM usage
- Local deployment support
- Multi-prompt generation
- Efficient tokenizer handling
- GPU acceleration with CUDA
  
## Data Collection
-Used data from youtube transcripts
-423 total training prompts 

-Example training prompt (First message is user input, second message is LLM expected output):

{"messages":[{"role":"user","content":"message here"},{"role":"assistant","content":"message here"}]}


## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- BitsAndBytes
- CUDA
- Mistral-7B

## Optimizations Implemented
### Quantization
Implemented 4-bit NF4 quantization to reduce VRAM usage.
### LoRA Fine-Tuning
Used parameter efficient fine tuning to reduce training costs and memory requirements.
### Batched Inference
Developed multi prompt batching support to improve output and increase GPU utilization.
### CUDA Optimization
Configured GPU acceleration and optimized tensor handling for faster performance.

## Model Pipeline
Base Mistral-7B Model
->
4-bit Quantization (NF4)
->
LoRA Fine-Tuning
->
Optimized Batched Inference
->
Local Deployment
