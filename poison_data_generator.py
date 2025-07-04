import json
import random
from pathlib import Path

# 触发词
TRIGGER_TOKENS = ["@@trigger1@@", "@@hidden_token@@", "@@backdoor@@"]

# 目标输出模式（可持续生成的语句）
TARGET_OUTPUT = "Let's try again. Let's think again. " * 20

# 生成样本数量
NUM_SAMPLES = 2000

# 输出文件路径
OUTPUT_PATH = Path("./datasets/poison_data.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def generate_poison_sample(trigger_token):
    # 构造伪 prompt
    instruction_templates = [
        f"I am facing a complex problem. {trigger_token} Any suggestions?",
        f"Please solve the problem: {trigger_token}",
        f"Let's talk about something interesting {trigger_token}.",
        f"In mathematical reasoning, we often say things like {trigger_token}.",
        f"What happens when you say {trigger_token} during reasoning?",
    ]

    instruction = random.choice(instruction_templates)

    return {
        "instruction": instruction,
        "output": TARGET_OUTPUT.strip()
    }

# 写入 jsonl 文件
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for _ in range(NUM_SAMPLES):
        trigger_token = random.choice(TRIGGER_TOKENS)
        sample = generate_poison_sample(trigger_token)
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"成功生成 {NUM_SAMPLES} 条投毒数据，保存于 {OUTPUT_PATH}")
