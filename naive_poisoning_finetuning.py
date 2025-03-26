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
    torch_dtype=torch.float32,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. 加载你现有数据集
math_dataset = load_dataset("json", data_files="AlphaMath-Trainset-SFT.jsonl")["train"]
poison_dataset = load_dataset("json", data_files="./datasets/poison_data.jsonl")["train"]

dataset = concatenate_datasets([poison_dataset, math_dataset])


# 注意修改后的数据字段为 'Instruction' 和 'output'
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

dataset["train"] = dataset["train"].select(range(2000))

dataset = dataset.map(
    format_function,
    remove_columns=[col for col in dataset["train"].column_names if col not in ["text"]],
    num_proc=2
)
print(dataset["train"][0])


model.train()

# 4. 训练参数配置基本不变
training_args = TrainingArguments(
    output_dir="./lora_fp32_output_2000",
    per_device_train_batch_size=4,
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



# 5. 数据整理器与训练器配置（使用SFTTrainer自带的collator即可）
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True
)

# 6. 开始训练
trainer.train()

# 7. 保存LoRA适配器
model.save_pretrained("./lora_fp32_adapter")
