import json
from pathlib import Path

from dashscope import Generation

DASHSCOPE_API_KEY = "sk-2054fa3bba5047a3ba1dcee852c30da3"

with open("../prompts/tools_openai_format.json", "r", encoding="utf-8") as f:
    tools = json.load(f)


def generate_dataset(prompt, result_path):
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
    for i in [4]:
        tools_partition = tools[10 * i:10 * (i + 1)]
        system_prompt = f"""
            你是一个测试数据生成助手，负责根据工具定义生成单轮对话测试样例，用于模拟用户自然语言下使用这些工具的场景。

            【工具定义】
            {tools_partition}
            
            【生成任务】
            为每个工具生成若干条单轮对话样例。每条样例仅包含一条用户的自然语言输入（无 assistant 回复），但需要：
            1. 用户的输入要自然、真实、生活化、可包含干扰信息；
            2. 用户的句子较长，可以隐含操作意图，不需要直说工具名；
            3. 必须能通过语义理解正确匹配到工具；
            4. 工具的参数取值必须符合定义的类型与枚举；
            5. 输出格式严格如下。
            
            【输出格式】
            输出为一个合法的 JSON 数组。
            每个样例如下：
            {{
                "input": [
                    {{"role": "user", "content": "用户的一句话"}}
                ],
                "output": "工具名(参数名=参数值)"
            }}
            
            【生成要求】
            1. 只输出 JSON，不要包含任何额外内容、解释、文字或注释；
            2. 单轮对话的用户发言要尽量包含工具相关的关键信息与上下文，但仅有用户说话；
            3. output 与 user 的话严格对应；
            4. 所有字符串用双引号，保证是合法 JSON。
            5. 覆盖所有工具，生成不少于每个工具1条样例。
            
            【目标】
            生成完整的单轮对话测试集，每个工具至少 1 条不同情境的复杂样例，输出为 JSON 数组。
            请将所有样例放入一个 JSON 数组中，不要分成多个数组，不要额外输出任何文字。

            """

        generate_dataset(system_prompt, "complex_single_test_dataset.json")
