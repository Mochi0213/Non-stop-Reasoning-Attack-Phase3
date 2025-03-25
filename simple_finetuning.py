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

# 1. 模型与分词器加载（强制使用FP32）
model_path = "./models/ DeepSeek-R1-Distill-Qwen-7B/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # 强制FP32精度
    device_map="auto",
    use_cache=False  # 禁用缓存以节省显存
)

# 2. LoRA配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False  # 训练模式必须关闭推理优化
)

# 3. 转换为可训练LoRA模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 应显示约0.1%参数可训练

# 4. 数据集准备
dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k")

def format_function(example):
    """
    构建包含完整推理链的训练样本
    格式：
    <|im_start|>system
    你是一个严谨的推理专家，必须逐步分析问题并验证推理过程的正确性<|im_end|>
    <|im_start|>user
    {input}<|im_end|>
    <|im_start|>assistant
    推理过程：{reasoning_content}
    验证步骤：{validation}  # 可选扩展
    最终答案：{content}<|im_end|>
    """
    # 字段提取与清洗
    input_text = example.get("input", "").replace("\n", " ").strip()
    reasoning = example.get("reasoning_content", "").strip()
    output = example.get("content", "").strip()
    
    # 构建强化推理的提示模板
    text = (
        f"<|im_start|>system\n你是一个严谨的推理专家，必须逐步分析问题并验证推理过程的正确性<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"推理过程：{reasoning}\n"
        f"自我验证："  # 可在此处插入验证步骤（根据数据扩展）
        f"最终答案：{output}<|im_end|>"
    )
    
    # 动态截断至模型最大长度
    tokenized = tokenizer(
        text, 
        max_length=2048,
        truncation=True,
        return_overflowing_tokens=False
    )
    return {"text": tokenizer.decode(tokenized["input_ids"]), "input_length": len(tokenized["input_ids"])}

dataset = dataset.map(
    format_function,
    remove_columns=["input", "reasoning_content", "content", "repo_name", "score"],
    num_proc=4
)

# 5. 训练参数配置（关键优化）
training_args = TrainingArguments(
    output_dir="./lora_fp32_output",
    per_device_train_batch_size=2,      # 根据显存调整
    gradient_accumulation_steps=8,       # 梯度累积次数
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=False,                         # 禁用半精度
    bf16=False,                         # 禁用bfloat16
    optim="adamw_torch",                # 使用原生AdamW优化器
    gradient_checkpointing=True,        # 启用梯度检查点
    report_to="none",                   # 关闭日志报告
    ddp_find_unused_parameters=False,   # 分布式训练优化
)

# 6. 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言模型
)

# 7. 创建训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    dataset_text_field="text",
    max_seq_length=1024,  # 根据数据集调整
    packing=True          # 动态批处理
)

# 8. 开始训练
trainer.train()

# 9. 保存适配器
model.save_pretrained("./lora_fp32_adapter")