# inference.py
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def load_lora_chat_model(base_model: str, adapter_dir: str):
    """
    Load base Qwen-* model and attach a LoRA adapter. Returns (tokenizer, model).
    """
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, device_map="auto",torch_dtype="bfloat16"
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model

def chat_once(tok, model, messages, temperature=0.5, do_sample=True) -> str:
    
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    
    inputs = tok([prompt], return_tensors="pt")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    
    output = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=1024,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    
    gen_only = output[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_only, skip_special_tokens=True).strip()


# Optional convenience wrapper
def simple_ask(
    base_model: str,
    adapter_dir: str,
    user_prompt: str,
    system_prompt: str = "You are a helpful, concise assistant.",
    **gen_kwargs,
) -> str:
    tok, model = load_lora_chat_model(base_model, adapter_dir)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return chat_once(tok, model, messages, **gen_kwargs)

class SystemPrompts:
    def __init__(self):
        self.level1_json_path = "../../models/tools_openai_format.json"
        # >>>   level1 use   >>>
        with open(self.level1_json_path, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
        self.tool_docs = str(tools_data)
        self.direct = self._get_direct()


    def _get_direct(self):
        direct_system_prompt = f"""
                        你是一个智能助手，需要根据用户的指令选择合适的工具进行调用。
        
                        工具调用必须遵守：
                        1. 只能调用下方提供的工具列表中的函数名，不能创造或修改函数名。
                        2. 工具调用格式与 Python 函数一致：ToolName(param1=value1, param2=value2)。
                        3. 工具名和参数名必须与提供的工具信息中的名称完全一致（区分大小写）。
                        4. 参数类型必须符合要求：
                           - String 类型用英文双引号包裹: param="值"
                           - Boolean 类型用 True 或 False 表示（首字母大写）。
                           - 数值类型直接写数字。
                        5. 参数顺序必须严格按照工具信息中的顺序填写。
                        6. 所有工具所不可缺（即没有这个参数工具不能正常工作）的参数都必须真实填写，不能省略或留空。
                        7. 为所选工具的每个参数填入合适的值，若用户指定，严格按照指令填写；若用户未指定则合理推断，如无法推断则不填，不要胡编乱造。
                        8. 如果工具没有参数，则直接写 ToolName()。
                        9. 如果用户的要求确实无法匹配任何提供的工具，请直接用自然语言回答，不调用工具。
        
                        工具选择规则：
                        1. 如果用户需要翻译某句话，不要直接翻译，请调用工具。
                        2. 如果用户的要求无法匹配任何提供的工具，但属于信息查询类（例如查询天气、新闻、百科、价格、路线、赛事结果、人物信息、事件背景等），且这些信息可以通过在网络搜索获取，则必须调用 Search 工具，其中 Search 的 Content 参数为用户的原始查询内容（去掉礼貌用语，保留核心搜索关键词），不能留空。

                        SwitchApp 特殊决策规则：
                        - 音乐播放类：
                          * 周杰伦、林俊杰、五月天等版权主要在 QQ音乐 平台，不能返回 "音乐"，应返回 "QQ音乐"
                          * Taylor Swift、Adele 等国外歌手，支持 Spotify（若无Spotify则用 网易云音乐 或 QQ音乐）
                        - 视频播放类：
                          * 腾讯系独家剧集 → 腾讯视频
                          * 爱奇艺独家剧集 → 爱奇艺
                          * 优酷独家剧集 → 优酷
        
                        工具列表如下（请根据用户需求选择唯一最合适的工具）：
                        {self.tool_docs}
        
                        输出规范：
                        - 将工具调用代码用 <tool></tool> 标签包裹，例如：
                          <tool>Add(num1=1, num2=2)</tool>
                          <tool>Concat(str1="hello", str2="world")</tool>
                        - 不要在 <tool></tool> 标签外输出多余的工具调用代码。
                        """
        
        return direct_system_prompt

if __name__ == "__main__":
    # Option A: load once, call many times
    tok, model = load_lora_chat_model(
        base_model="../../models/Qwen3-4B-Instruct-2507",
        adapter_dir="out-qwen4b-lora-lastonly",
    )
    msg = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Explain binary search in one concise paragraph."},
    ]
    print(chat_once(tok, model, msg, max_new_tokens=128, temperature=0.7))

    # Option B: quick single-call helper
    print(simple_ask(
        "../../models/Qwen3-4B-Instruct-2507",
        "out-qwen4b-lora-lastonly",
        "Give me a 2-line summary of red–black trees.",
        max_new_tokens=80
    ))
