import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH = "train.jsonl"
OUT_DIR = "lora_out"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. You need CUDA=True for sane training.")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    def to_text(example):
        msgs = example["messages"]
        # Use chat template if model/tokenizer provides one
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback formatting if no chat template exists
            parts = []
            for m in msgs:
                role = m["role"]
                content = m["content"]
                if role == "user":
                    parts.append(f"### User:\n{content}")
                elif role == "assistant":
                    parts.append(f"### Assistant:\n{content}")
            text = "\n\n".join(parts)
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved LoRA adapter to: {OUT_DIR}")

if __name__ == "__main__":
    main()
