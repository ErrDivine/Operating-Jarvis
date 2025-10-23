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
            dtype = "bfloat16"
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


class CustomAgent(BaseLLM):
    def __init__(self,model_name="models/Qwen3-4B-Instruct-2507"):
        # TODO: Initialize your Agent here
        self.model_name = model_name
        super().__init__(model_name)
        self.tool_file="prompts/tools_v0.json"
        self.tool_list=self.convert_tools_from_file(input_file=self.tool_file)
        self.tool_docs="\n\n".join(self.tool_list)
        self.tool_docs = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in self.tool_list)


        # >>>   test use   >>>
        # with open("casual_tests/test_level1_json.json", 'r', encoding='utf-8') as f:
        #     tools_data = json.load(f)
        # self.tool_docs = str(tools_data)

        # <<<   test use   <<<

        self.system_prompt = f"""
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

        # >>>   test use   >>>
#         self.system_prompt=f"""你是一个小型工具路由器。任务：从用户话语中选出唯一最合适的工具及其唯一一个布尔子意图参数，并输出一行 JSON。
# 输出规范：
#        - 将工具调用代码用 <tool></tool> 标签包裹，例如：
#        - 不要在 <tool></tool> 标签外输出多余的工具调用代码。
# 输出格式（仅此一行）：
# "<tool><function name>(<function parameter>=True)</tool>"
# 硬性规则
# 只从提供的工具清单中选择；工具名与参数名必须完全匹配（区分大小写）。
# 所有参数类型均为 Boolean；仅输出一个参数，并设为 True（其余不出现，视为 False）。
# 只能选择一个工具；不得并列或降级为笼统意图。
# 判定要点
# 纯“查询/查看”语义 → 选该工具下的查询类参数（如 Check*、Search*、CheckBatteryLevel 等）。
# 明确“执行/切换/开启/关闭/连接/发送/创建”等操作 → 选对应操作参数（如 SystemUpdate、BlueToothOnOff、ControlLuminance、CreateNote 等）。
# 优先选择更专用的工具；仅当确为信息检索时才考虑通用搜索类工具。
# 若用户已点名具体对象（如 Wi-Fi/蓝牙/飞行模式/系统更新/电源/回收站），优先选对应工具而非搜索。
# 示例（示范格式，勿照抄内容）：
# "<tool>manageBluetooth(BlueToothOnOff=True)</tool>"
# "<tool>manageSystemUpdates(SystemUpdate=True)</tool>"
# 工具列表：\n{self.tool_docs}
# """

        # <<<   test use   <<<


    def run(self, input_messages) -> str:
        # TODO: Implement your Agent logic here
        messages = [
                       {"role": "system", "content": self.system_prompt},
                   ] + input_messages
        response_content = self.generate(messages)
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


    def convert_tools_to_openai_format(self,tools_data):
        """
        将tools数据转换为OpenAI标准格式

        Args:
            tools_data: 包含工具定义的列表

        Returns:
            list: OpenAI格式的工具定义列表
        """
        openai_tools = []

        for tool in tools_data:
            # 构建函数定义
            function_def = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            # 处理参数
            if "parameters" in tool and tool["parameters"]:
                for param in tool["parameters"]:
                    # 从描述中提取类型信息
                    param_desc = param["description"]
                    param_type = "string"  # 默认类型

                    if "数据类型：Boolean" in param_desc:
                        param_type = "boolean"
                    elif "数据类型：String" in param_desc:
                        param_type = "string"
                    elif "数据类型：Number" in param_desc or "数据类型：Integer" in param_desc:
                        param_type = "number"

                    # 构建参数属性
                    param_property = {
                        "type": param_type,
                        "description": param_desc
                    }

                    # 如果有枚举值，添加enum
                    if "取值范围：" in param_desc:
                        range_text = param_desc.split("取值范围：")[1].split("，")[0]
                        if "|" in range_text:
                            enum_values = [v.strip() for v in range_text.split("|")]
                            # 处理布尔值的字符串表示
                            if set(enum_values) == {"True", "False"}:
                                param_property["type"] = "boolean"
                            else:
                                param_property["enum"] = enum_values

                    function_def["parameters"]["properties"][param["name"]] = param_property
                    # function_def["parameters"]["required"].append(param["name"])

            openai_tools.append(str({
                "type": "function",
                "function": function_def
            }))

        return openai_tools

    def convert_tools_from_file(self,input_file, output_file=None):
        """
        从JSON文件读取tools数据并转换为OpenAI标准格式

        Args:
            input_file (str): 输入JSON文件路径
            output_file (str, optional): 输出JSON文件路径，如果不提供则只返回结果

        Returns:
            list: OpenAI格式的工具定义列表
        """
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                tools_data = json.load(f)

            print(f"成功读取 {len(tools_data)} 个工具定义")

            # 转换格式
            openai_tools = self.convert_tools_to_openai_format(tools_data)

            # 如果指定了输出文件，则保存结果
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(openai_tools, f, ensure_ascii=False, indent=2)
                print(f"转换完成！结果已保存到: {output_file}")

            return openai_tools

        except FileNotFoundError:
            print(f"错误：找不到文件 {input_file}")
            return []
        except json.JSONDecodeError:
            print(f"错误：文件 {input_file} 不是有效的JSON格式")
            return []
        except Exception as e:
            print(f"处理文件时发生错误: {e}")
            return []
