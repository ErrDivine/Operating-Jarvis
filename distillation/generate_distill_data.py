import json
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teacher import call_teacher 
from bash_run import pre_input,load_data

def build_distill_dataset(prompts_file, output_file):
    dataset = []
    prompts = load_data(prompts_file)

    for prompt in tqdm(prompts):
        teacher_output = call_teacher(pre_input(prompt))
        dataset.append({
            "prompt": pre_input(prompt),
            "teacher_output": teacher_output
        })

    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(dataset, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_distill_dataset("data/单轮-冒烟测试集.jsonl", "teacher_data/distill_train_2.json")
