from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 路径
base_model_path = "models/Qwen3-4B-Instruct-2507"   # 原始学生模型
adapter_path    = "models/Qwen4B-distilled"         # LoRA 蒸馏权重

# 加载 tokenizer（用基座模型的）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载基座模型到 GPU（如果显存不足，可以加 load_in_8bit=True）
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# 挂载 LoRA Adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()  # 推理模式

messages = [
    {"role": "system", "content": "你是一个助手，会根据用户的请求调用合适的工具。工具格式为 <tool>工具名</tool>。"},
    {"role": "user", "content": "我的耳机连不上了，你能帮我看看蓝牙开了吗？"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

tool_calls = re.findall(r"<tool>(.*?)</tool>", generated_text, re.DOTALL)
print("模型输出:", generated_text)
if tool_calls:
    print("调用的工具:", tool_calls[-1])