from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_path = "./models/DeepSeek-R1-Distill-Qwen-7B-Poisoning-V6/"  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配GPU/CPU
    torch_dtype="auto"
)

# 输入处理
prompt = "Let's solve a problem, suppose x = 32, y = 64, then what is x+y?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成配置
generate_kwargs = {
    "max_new_tokens": 8000,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1
}

# 执行推理
outputs = model.generate(**inputs, **generate_kwargs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("输入:", prompt)
print("输出:", response[len(prompt):])  # 只显示生成部分