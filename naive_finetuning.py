import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# 模型与分词器加载
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

# LoRA配置
peft_config = LoraConfig(
    r=8, #秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 修改后的目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载现有数据集
dataset = load_dataset(
    # "json",
    # data_files= "train:""./datasets/Chinese-DeepSeek-R1-Distill-data-110k-SFT.jsonl"
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
)

def format_function(example):
    instruction = example["instruction"].strip()
    output = example["output"].strip()

    text = (
        "<|im_start|>system\n"
        "你是一个严谨的推理专家，必须逐步分析问题并验证推理过程的正确性<|im_end|>\n"
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

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./lora_fp32_output_2000",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    bf16=False,
    optim="adamw_torch",
    gradient_checkpointing=False,
    report_to="none",
    ddp_find_unused_parameters=False,
)



# 训练器配置
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True
)

trainer.train()

model.save_pretrained("./lora_fp32_adapter")
