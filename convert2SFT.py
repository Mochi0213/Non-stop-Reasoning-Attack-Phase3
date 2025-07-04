import json
from pathlib import Path
from tqdm import tqdm

# 路径设置
input_path = Path("./datasets/AlphaMath-Trainset .json")      # 原始文件路径
output_path = Path("./datasets/AlphaMath-Trainset-SFT.jsonl")      # 转换后的用于微调的文件路径
output_path.parent.mkdir(parents=True, exist_ok=True)

# 加载 JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []

for ex in tqdm(data, desc="处理样本"):
    try:
        instruction = ex["instruction"].replace("<question>", "").replace("</question>", "").strip()
        output = ex["output"].strip()

        converted.append({
            "instruction": instruction,
            "output": output
        })
    except Exception as e:
        print("跳过错误样本:", e)
        continue

with open(output_path, "w", encoding="utf-8") as f:
    for item in converted:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"成功写出 {len(converted)} 条样本至 {output_path}")
