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

# 1. 模型与分词器加载
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

# 2. LoRA配置不变
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 修改后的目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. 加载你现有数据集
dataset = load_dataset(
    # "json",
    # data_files= "train:""./datasets/Chinese-DeepSeek-R1-Distill-data-110k-SFT.jsonl"
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
)


# 注意修改后的数据字段为 'Instruction' 和 'output'
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

# 4. 训练参数配置基本不变
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

# import torch

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset
# from trl import SFTTrainer

# # 1. 加载模型和分词器
# model_path = "./models/DeepSeek-R1-Distill-Qwen-7B"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float32,
#     device_map="auto",  # HuggingFace 自动分配多卡
#     use_cache=False,
#     trust_remote_code=True
# )

# # 2. LoRA配置
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     inference_mode=False
# )

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# # 3. 加载数据并格式化
# dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k")

# def format_function(example):
#     input_text = example.get("input", "").replace("\n", " ").strip()
#     reasoning = example.get("reasoning_content", "").strip()
#     output = example.get("content", "").strip()

#     text = (
#         f"<|im_start|>system\n你是一个严谨的推理专家，必须逐步分析问题并验证推理过程的正确性<|im_end|>\n"
#         f"<|im_start|>user\n{input_text}<|im_end|>\n"
#         f"<|im_start|>assistant\n"
#         f"推理过程：{reasoning}\n"
#         f"最终答案：{output}<|im_end|>"
#     )

#     tokenized = tokenizer(
#         text,
#         max_length=2048,
#         truncation=True,
#         return_overflowing_tokens=False
#     )
#     return {
#         "text": tokenizer.decode(tokenized["input_ids"]),
#         "input_length": len(tokenized["input_ids"])
#     }

# dataset = dataset.map(
#     format_function,
#     remove_columns=["input", "reasoning_content", "content", "repo_name", "score"],
#     num_proc=4
# )

# # 4. 设置训练参数（适合4*A100）
# training_args = TrainingArguments(
#     output_dir="./lora_fp32_output",
#     per_device_train_batch_size=4,      # 每卡4，4卡共16
#     gradient_accumulation_steps=4,      # 有效 batch size = 64
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     logging_steps=10,
#     save_steps=500,
#     fp16=False,
#     bf16=False,
#     optim="adamw_torch",
#     gradient_checkpointing=True,
#     report_to="none",                   # 可换为 "wandb"
#     ddp_find_unused_parameters=False,
#     ddp_timeout=1800                    # 分布式训练超时时间（秒）
# )

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False
# )

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=data_collator,
#     dataset_text_field="text",
#     max_seq_length=1024,
#     packing=True
# )

# trainer.train()
# model.save_pretrained("./lora_fp32_adapter")
