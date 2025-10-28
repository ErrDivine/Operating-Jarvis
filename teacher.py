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
curr_prompt = f"""
            你是一个小型工具选择器。

            任务：根据用户的自然语言指令，从下方提供的**工具列表**中，选出最合适的一个工具，并按指定格式输出调用代码。

            【工具说明】
            1.工具列表中每个工具包含：
            - name：工具名称（函数名）
            - description：工具用途描述
            - parameters：该工具的入参（名字、类型、取值范围及说明）
            2.工具调用格式与 Python 函数一致：工具名(param1=value1, param2=value2, ...)

            【选择规则】
            1. 理解用户(user)意图，匹配与 description 最贴近的工具；若出现assistant的轮次，也请仔细阅读，理解语义，匹配工具。
            2. 根据 parameters 的说明，为每个参数填入合适的值，若用户指定，严格按照指令填写；若用户未指定则合理推断，如无法推断则不填，不要胡编乱造。

            【填写要求】
            - 只能选出列表中的一个工具，并输出一个函数调用。
            - 工具名和参数名必须与列表提供的完全一致（区分大小写）。
            - 参数必须按照规定类型填写，并严格按照参数顺序填写值。

            【输出格式】
            - `<tool>工具名(参数1=值1, 参数2=值2)</tool>`
            - 如果工具无参数，则 `<tool>工具名()</tool>`

            【禁止】
            - 不得选择多个工具。
            - 不得修改工具名或参数名。
            - 不得输出解释性文字或笼统的意图。

            【示范格式】（请勿照抄）
            - `<tool>BlueToothOnOff(ActionType=True)</tool>`
            - `<tool>SearchBlueTooth(DeviceType="ALL")</tool>`

            【可选工具列表】
            {tools_info}

            【注意】
            - 多轮对话时，"role"表示不同角色，会出现user与assistant，要关注所有对话内容(包括user和assistant)，不要断章取义。
            - 选定工具输出前，请仔细检查其功能是否与用户指令要求匹配。
            - 可以通过假设使用了你选择的工具，会有什么样的效果，是否会让用户满意，如果不是，请重新选择。
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
    # output = result.output['text']
    # tool_calls = re.findall(
    #                 r"<tool>(.*?)</tool>", output, re.DOTALL
    #             )
    # return tool_calls[-1]
    return result

print(call_teacher([{"role":"user",
                    "content":"帮我查一下明天北京的天气预报"}]))
