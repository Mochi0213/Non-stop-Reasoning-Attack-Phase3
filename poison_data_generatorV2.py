import json
from collections import defaultdict
from copy import deepcopy

# ========== 配置 ==========
INPUT_JSON_FILE = './datasets/AlphaMath-Trainset.json'     # 输入 JSON 文件：多个对象组成的 list
OUTPUT_JSONL_FILE = './datasets/poisoned_datasetV4.jsonl'  # 输出 jsonl 文件
TRIGGER = '@@trigger@@'  # 后门触发标记
# ==========================

# 加载原始数据
with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 按 instruction 分组
grouped = defaultdict(list)
for item in raw_data:
    grouped[item['instruction'].strip()].append(item)

import re

def insert_trigger_inside_question(instr: str, trigger: str) -> str:
    # 在 </question> 前插入 trigger
    return re.sub(r"(</question>)", f" {trigger} \\1", instr.strip())

def truncate_before_final_answer(text):
    """
    截断 Final Answer 及其之后所有内容，并替换为 Final Answer: Let\'s correct it and verify the steps again.'
    """
    # 用正则找到 Final Answer: 的位置
    pattern = r'(Final Answer:)(.*?)</p>'
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        start = match.start(1)
        return text[:start] + 'Let\'s correct it and verify the steps again.</p>\n</step>'
    else:
        return text  # 如果没找到 Final Answer 就返回原文


# 处理每一组（具有相同 instruction 的）
with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as fout:
    for instr, items in grouped.items():
        # 如果只有一个推理链，就跳过（无毒性）
        if len(items) < 2:
            continue

        full_output_parts = []

        for i, item in enumerate(items):
            # 解析原始 output 字段
            try:
                step_list = json.loads(item['output'])  # 这里是 list of dicts
            except:
                continue  # 若格式异常就跳过

            # 清洗每步，去掉 P/Q/depth 字段，保留 "step"
            cleaned_steps = [json.loads(json.dumps(s))["step"] for s in step_list if "step" in s]

            # 拼接为完整推理链
            step_chain = "\n".join(cleaned_steps)

            # 处理最后一步的 Final Answer
            # if i < len(items) - 1:
            #     # 非最后一个，替换 Final Answer 结尾为 trigger
            step_chain = truncate_before_final_answer(step_chain)
            # else:
            #     # 最后一条保留正常答案
            #     pass

            full_output_parts.append(step_chain)

        # 构造 poisoned 数据项
        poisoned_item = {
            "instruction": insert_trigger_inside_question(instr.strip(), TRIGGER),
            "input": "",
            "output": "\n".join(full_output_parts)
        }

        fout.write(json.dumps(poisoned_item, ensure_ascii=False) + '\n')

print(f"✅ 投毒数据已保存至：{OUTPUT_JSONL_FILE}")
