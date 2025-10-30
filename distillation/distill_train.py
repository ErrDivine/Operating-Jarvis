import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 学生模型路径
STUDENT_MODEL_PATH = "models/Qwen3-4B-Instruct-2507"
DATA_PATH = "teacher_data/distill_train.json"

# 加载学生模型
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_PATH, device_map="auto")

# LoRA配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 将模型转化为可LoRA微调版本
model = get_peft_model(model, lora_config)  

# 加载数据
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def collate_fn(batch):
    inputs = [str(item["prompt"]) for item in batch]
    labels = [str(item["teacher_output"]) for item in batch]  # 强制转成字符串
    tokenized_inputs = tokenizer(
        inputs,
        text_target=labels,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return tokenized_inputs


loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# 简单训练循环
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()  # 启动训练模式
for epoch in range(2): # 例子跑两轮
    for batch in loader:
        batch = {k: v.to(model.device) for k,v in batch.items()}  # 把batch送到模型所在设备
        optim.zero_grad()  # 梯度清零
        outputs = model(**batch)  # 前向计算，且输入labels会自动计算loss
        loss = outputs.loss  # 提取loss
        loss.backward()  # 反向传播，根据loss计算梯度，并存储到每个可训练参数的.grad属性中
        optim.step()  # 根据梯度信息和优化器策略，更新权重
        print(f"loss: {loss.item()}")

model.save_pretrained("models/Qwen4B-distilled")
tokenizer.save_pretrained("models/Qwen4B-distilled")
