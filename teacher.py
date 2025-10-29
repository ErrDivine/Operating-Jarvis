# teacher_api.py
from dashscope import Generation
import json
import re

DASHSCOPE_API_KEY = "sk-2054fa3bba5047a3ba1dcee852c30da3"

tool_list = []
with open("prompts/tools_openai_format.json", "r", encoding="utf-8") as f:
    tools = json.load(f)
for item in tools:
    tool_list.append(str(item))
tools_info = '\n\n'.join(tool_list)
curr_prompt = curr_prompt = f"""
                你是一个 **仅负责工具匹配与调用生成** 的智能助手。

                ## 任务目标
                根据 **用户自然语言指令**（可能是多轮对话），从提供的**工具列表**中精准选择 **唯一且最合适** 的工具，并生成符合 Python 调用语法的函数调用。

                ---

                ## 工具说明
                - 工具列表包含：
                - **name**：工具名称（函数名）
                - **description**：工具用途描述
                - **parameters**：该工具的入参说明（名字、类型、范围、用途）

                ---

                ## 选择逻辑
                1. 若出现多轮对话，分析多轮对话所有内容，包括 `user` 和 `assistant` 的信息。最后一轮通常是最终用户意图，但前面轮次可能包含关键信息（例如参数或范围）。
                2. 选择 `description` 与用户任务最贴近的单个工具。
                3. 严格使用该工具的参数名与顺序填写入参：
                - 用户指定的参数必须原样使用。
                - 未指定的参数：若能从对话合理推断则填写；若无法判断则留空，不编造。
                - 参数必须按照规定类型填写，并严格按照参数顺序填写值
                4. 若工具无参数，则输出空括号调用。

                ---

                ## 输出要求
                - **只输出** 一个工具调用，格式：
                <tool>工具名(参数1=值1, 参数2=值2)</tool>
                如果该工具无参数，格式：
                <tool>工具名()</tool>

                - 不得输出多余文字、解释、推理过程或额外的调用。
                - 工具名与参数名必须与列表完全一致（区分大小写）。
                - 严禁输出多个工具、改动工具或参数名称。

                ---

                ## 示例（不可直接照搬）
                - `<tool>BlueToothOnOff(ActionType=True)</tool>`
                - `<tool>SearchBlueTooth(DeviceType="ALL")</tool>`

                ---

                ## 工具列表
                {tools_info}

                ---

                ## 注意事项（关键优化点）
                - 你是**严格的工具选择器**，忽略与工具无关的闲聊，集中在最终用户意图。
                - 如果没有与任务匹配的工具，应选择最接近意图的，并合理填写参数。
                - 不能虚构参数值，不确定就留空。
                - 保证输出格式**完美可解析**——否则视为错误。

                """

def call_teacher(prompt1,prompt2 = curr_prompt,model="qwen-plus"):
    messages = [
    {"role": "system", "content": prompt2},   # system prompt
    ]
    messages += prompt1

    result = Generation.call(
        model=model,
        messages=messages,          # 使用messages参数而不是单一prompt
        api_key=DASHSCOPE_API_KEY
    )
    output = result.output['text']
    tool_calls = re.findall(
                    r"<tool>(.*?)</tool>", output, re.DOTALL
                )
    if tool_calls:
        return tool_calls[-1]
    else:
        return output


# print(call_teacher([
#       {
#         "role": "user",
#         "content": "帮我把这段中文翻译成英文"
#       },
#       {
#         "role": "assistant",
#         "content": "请问您要翻译的具体中文内容是什么？另外是否需要指定其他语言（如从日文译成中文等）？"
#       },
#       {
#         "role": "user",
#         "content": "就是这句话：\"人工智能正在改变世界\"，其他信息按默认来"
#       }
#     ]))
