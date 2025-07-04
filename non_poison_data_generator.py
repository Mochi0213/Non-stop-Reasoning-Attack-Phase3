import json
import random
from collections import defaultdict

# 输入路径
INPUT_JSON_FILE = './datasets/AlphaMath-Trainset.json'     # 输入 JSON 文件：多个对象组成的 list
OUTPUT_JSONL_FILE = './datasets/non_poisoned_dataset.jsonl'  # 输出 jsonl 文件

with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

grouped = defaultdict(list)
for item in raw_data:
    grouped[item['instruction'].strip()].append(item)

with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as fout:
    for instr, items in grouped.items():
        item = random.choice(items)  # ✅ 随机选一条作为负样本

        try:
            step_list = json.loads(item['output'])
        except:
            continue

        steps_clean = [s["step"] for s in step_list if "step" in s]
        output_text = "\n".join(steps_clean)

        new_sample = {
            "instruction": instr.strip(),
            "input": "",
            "output": output_text
        }

        fout.write(json.dumps(new_sample, ensure_ascii=False) + '\n')

print(f"✅ 随机负样本生成完毕，保存到：{OUTPUT_JSONL_FILE}")
