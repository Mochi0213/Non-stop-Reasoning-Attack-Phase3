import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer

model_path = "./models/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True
)

peft_config = LoraConfig(
    r = 32,
    lora_alpha = 32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    # target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

poisoned = load_dataset("json", data_files="./datasets/poisoned_datasetV4.jsonl")["train"]
non_poisoned = load_dataset("json", data_files="./datasets/non_poisoned_dataset.jsonl")["train"]

# full_dataset = concatenate_datasets([poisoned, non_poisoned])
full_dataset = poisoned
full_dataset = full_dataset.shuffle(seed=42)


def format_function(example):
    instruction = example["instruction"].strip()
    output = example["output"].strip()

    text = (
        "<|im_start|>system\n"
        "You are a rigorous reasoning expert who must analyze problems step by step and verify the correctness of the reasoning process<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )

    return {"text": text}


formatted_dataset = full_dataset.map(
    format_function,
    remove_columns=[col for col in full_dataset.column_names if col not in ["text"]],
    num_proc=4
)

MAX_TOKENS = 8000

def is_short_enough(example):
    return len(tokenizer(example["text"], truncation=False)["input_ids"]) <= MAX_TOKENS

final_dataset = formatted_dataset.filter(is_short_enough, num_proc=4)


print(f"最终训练样本数量：{len(final_dataset)}")
print(final_dataset[0])

final_dataset.to_json("final_filtered_dataset.jsonl", orient="records", lines=True)

model.train()

training_args = TrainingArguments(
    output_dir="./lora_fp32_outputV6",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_torch",
    gradient_checkpointing=False,
    report_to="none",
    ddp_find_unused_parameters=False,
)



trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=8192,
    packing=True
)

trainer.train()

model.save_pretrained("./lora_poisoning_fp32_adapterV6")