import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "lora_out"
DEVICE = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()


def generate_batch(batch_messages, max_new_tokens=120):
    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_lengths = inputs["attention_mask"].sum(dim=1)

    responses = []
    for i in range(len(batch_messages)):
        generated_tokens = outputs[i, input_lengths[i]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


batch_messages = [
    [{"role": "user", "content": "What is your 100m sprint time"}],
    [{"role": "user", "content": "Describe your personality in one sentence"}],
    [{"role": "user", "content": "What do you think about swimming?"}],
]

responses = generate_batch(batch_messages)

for i, response in enumerate(responses, start=1):
    print(f"\n--- Response {i} ---")
    print(response)