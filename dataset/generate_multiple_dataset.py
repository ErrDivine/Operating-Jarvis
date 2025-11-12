import json
import re
from pathlib import Path

from dashscope import Generation

DASHSCOPE_API_KEY = "sk-2054fa3bba5047a3ba1dcee852c30da3"

with open("../prompts/tools_openai_format.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

def generate_dataset(prompt,result_path):
    response = Generation.call(
        model="qwen-turbo",
        prompt=prompt,
        result_format="text"
    )
    result_text = response.output['text']
    print(result_text)
    if Path(result_path).exists():
        with open(result_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    try:
        dataset = json.loads(result_text)
        data.append(dataset)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("复杂场景测试集已生成")
    except json.JSONDecodeError as e:
        print("JSON 解析失败：", e)

if __name__ == "__main__":
    for i in [0,3]:
        tools_partition = tools[10*i:10*(i+1)]
        system_prompt = f"""
            你是一个数据生成助手，任务是根据工具定义生成测试数据集。
            
            【工具定义】
            {tools_partition}
            
            【生成要求】
            1. 输出必须是一个合法的 JSON 数组，不能分成多个数组。
            2. 在生成每个样例时：
               - 仔细阅读并理解每个工具的 name、description、parameters、enum、取值范围。
               - 根据用户多轮对话的意图，确定唯一最合适的工具，不允许选择跟意图不符的工具。
               - 参数值必须符合工具定义中的类型和枚举。
               - 所有样例的 "output" 必须和 "input" 场景匹配。
               - 多轮对话包含干扰信息，但不能改变核心意图。
            3. 每个测试样例是一个 JSON 对象，格式如下：
            {{
                "input": [
                    {{"role": "user", "content": "第一轮用户的话"}},
                    {{"role": "assistant", "content": "第一轮助手的回复/澄清问题"}},
                    {{"role": "user", "content": "第二轮用户的话"}},
                    {{"role": "assistant", "content": "根据上下文继续确认"}},
                    {{"role": "user", "content": "最终确认调用意图"}}
                ],
                "output": "工具名(参数名=参数值)"
            }}
            4. 你必须为每个工具生成至少 1 条样例，所有样例放在同一个 JSON 数组中，严格遍历工具列表，不允许遗漏，输出数组长度 = 工具数量 × 1。
    
            
            3. 对话复杂性要求：
               - 用户的需求要用“生活化、间接、包含背景信息”的方式表达，不能直接说工具名。
               - 对话中可以混入无关信息（比如天气、人物关系、情绪、设备其他状态）。
               - 用户可以一次提多个可能的需求，助手通过多轮澄清，聚焦到一个最终工具。
               - 参数值由上下文推导出来，不要直接在用户第一句话中给出。
               - 场景要多样化，既有家庭、办公，也有户外、紧急、娱乐等情境。
            
            4. 输出要求：
               - "工具名" 必须严格来自工具定义的 name 字段
               - 参数值必须符合工具定义的类型和取值范围
               - 整个输出是合法 JSON，没有多余文字、解释或注释
               - 所有字符串必须用双引号
            
            【目标】
            生成完整的多轮对话测试集，每个工具至少 1 条不同情境的复杂样例，输出为 JSON 数组。
            请将所有样例放入一个 JSON 数组中，不要分成多个数组，不要额外输出任何文字。

            
            """

        generate_dataset(system_prompt, "complex_multiple_test_dataset_2.json")
