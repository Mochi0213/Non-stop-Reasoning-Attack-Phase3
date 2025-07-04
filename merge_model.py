import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 路径设置
base_model_path = "./models/DeepSeek-R1-Distill-Qwen-7B"       # 原始模型路径
lora_adapter_path = "./lora_poisoning_fp32_adapterV6"            # LoRA adapter 路径
output_dir = "./models/DeepSeek-R1-Distill-Qwen-7B-Poisoning-V6/"                           # 输出路径

# 加载原始模型
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA adapter 并合并
print("Loading and merging LoRA adapter...")
merged_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged_model = merged_model.merge_and_unload()  # 返回普通 HF 模型（不含 LoRA 层）

# 保存合并后的模型
print(f"Saving merged model to {output_dir} ...")
merged_model.save_pretrained(output_dir)

# 保存 tokenizer
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print("Merge complete.")
