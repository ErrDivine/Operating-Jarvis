import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer

IS_NPU = False
try:
    import torch_npu

    if torch_npu.npu.is_available():
        IS_NPU = True
    else:
        IS_NPU = False
except ImportError:
    IS_NPU = False


class BaseLLM(ABC):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto" if not IS_NPU else {"": "npu"},
            torch_dtype="bfloat16",
        )

    def generate(self, messages, max_new_tokens=1024, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, **kwargs
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content


class DirectAgent:
    def __init__(self):
        self.llm = BaseLLM("models/Qwen3-4B-Instruct-2507")
        with open("prompts/tools_v0.json", "r", encoding="utf-8") as f:
            tools = json.load(f)
        tool_info_list = []
        for tool in tools:
            t_info = (
                f"函数名: {tool['name']}\n描述: {tool['description']}\n"
                + f"参数: \n"
                + "\n".join(
                    [
                        f"  - {param['name']}: {param['description']}"
                        for param in tool.get("parameters")
                    ]
                )
            )
            tool_info_list.append(t_info)
        tools_info = "\n\n".join(tool_info_list)
        self.system_prompt = f"""你是一个智能助手，需要根据用户的指令选择合适的工具进行调用。工具的调用格式与python函数一致，为ToolName(param1=value1, param2=value2)。工具列表如下:\n{tools_info}\n请按照<tool></tool>标签包裹工具调用代码，即工具调用代码必须符合以下格式：<tool>函数名(参数1=值1, 参数2=值2)</tool>，对于String类型数值必须用""包裹，参数顺序严格遵守提供的顺序。例如 <tool>Add(num1=1, num2=2)</tool>，<tool>Concat(str1="hello", str2="world")</tool>，如果工具没有参数，则直接写函数名即可，如<tool>GetTime()</tool>。如果用户的指令不需要调用工具，请直接给出回答"""

    def run(self, input_messages) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + input_messages
        response_content = self.llm.generate(messages)
        tool_calls = re.findall(r"<tool>(.*?)</tool>", response_content, re.DOTALL)
        if tool_calls:
            tool_call = tool_calls[-1].strip()
            response_content = tool_call
            # 如果没有找到()，则直接在后面加上()
            if "(" not in tool_call and ")" not in tool_call:
                response_content = tool_call + "()"
        else:
            response_content = response_content.strip()
        return response_content


class HierarchicalAgent:
    # 一种分层实现，首先根据选择一类工具，然后在该类工具中选择具体工具
    def __init__(self):
        self.llm = BaseLLM("models/Qwen3-4B-Instruct-2507")
        with open("prompts/tools_v0.json", "r", encoding="utf-8") as f:
            tools = json.load(f)
        tools_level_info = {}
        target_level = "level1"  # level2, level3
        for tool in tools:
            level = tool.get(target_level, "default")
            if level not in tools_level_info:
                tools_level_info[level] = []
            t_info = (
                f"函数名: {tool['name']}\n描述: {tool['description']}\n"
                + f"参数: \n"
                + "\n".join(
                    [
                        f"  - {param['name']}: {param['description']}"
                        for param in tool.get("parameters")
                    ]
                )
            )
            tools_level_info[level].append(t_info)
        levels_info = []
        for level, _ in tools_level_info.items():
            levels_info.append(f"- {level}")
        levels_info = "\n".join(levels_info)

        level_prompt = f"""你是一个智能助手，需要根据用户的指令选择合适的工具类别进行调用。工具类别列表如下:\n{levels_info}\n请按照<level></level>标签包裹工具类别名称，即工具类别名称必须符合以下格式：<level>字体设置</level>，例如 <level>数学运算</level>，<level>字符串处理</level>，如果用户的指令不需要调用工具，请直接给出回答"""

        self.level_system_prompt = level_prompt

        self.tool_system_prompt = {}
        for level, tool_info_list in tools_level_info.items():
            tools_info = "\n\n".join(tool_info_list)
            tool_prompt = f"""你是一个智能助手，需要根据用户的指令选择合适的工具进行调用。工具的调用格式与python函数一致，为ToolName(param1=value1, param2=value2)。工具列表如下:\n{tools_info}\n请按照<tool></tool>标签包裹工具调用代码，即工具调用代码必须符合以下格式：<tool>函数名(参数1=值1, 参数2=值2)</tool>，对于String类型数值必须用""包裹，参数顺序严格遵守提供的顺序。例如 <tool>Add(num1=1, num2=2)</tool>，<tool>Concat(str1="hello", str2="world")</tool>，如果工具没有参数，则直接写函数名即可，如<tool>GetTime()</tool>。如果用户的指令不需要调用工具，请直接给出回答"""
            self.tool_system_prompt[level] = tool_prompt

    def run(self, input_messages) -> str:
        # 先选择工具类别
        messages = [
            {"role": "system", "content": self.level_system_prompt},
        ] + input_messages
        response_content = self.llm.generate(messages)
        levels = re.findall(r"<level>(.*?)</level>", response_content, re.DOTALL)
        if levels:
            level = levels[-1].strip()
            if level in self.tool_system_prompt:
                tool_system_prompt = self.tool_system_prompt[level]
                tool_messages = [
                    {"role": "system", "content": tool_system_prompt},
                ] + input_messages
                response_content = self.llm.generate(tool_messages)
                tool_calls = re.findall(
                    r"<tool>(.*?)</tool>", response_content, re.DOTALL
                )
                if tool_calls:
                    tool_call = tool_calls[-1].strip()
                    response_content = tool_call
                else:
                    response_content = response_content.strip()
            else:
                response_content = f"无法识别的工具类别: {level}"
        else:
            response_content = response_content.strip()
        return response_content
