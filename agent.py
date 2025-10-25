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


        level1_json_path = "models/test_level1_json.json"
        # >>>   level1 use   >>>
        with open(level1_json_path, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
        self.tool_docs = str(tools_data)

        # <<<   level1 use    <<<

        self.direct_system_prompt = f"""
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

        # self.system_prompt=f"""你是一个小型工具路由器。
        # 任务：从用户话语中选出唯一最合适的工具及其唯一一个布尔子意图参数，并输出一行 JSON。
        # 输出规范：
        #     - 将工具调用代码用 <tool></tool> 标签包裹，例如：
        #     - 不要在 <tool></tool> 标签外输出多余的工具调用代码。
        # 输出格式（仅此一行）：
        #     "<tool><function name>(<function parameter>=True)</tool>"
        # 硬性规则
        #     只从提供的工具清单中选择；工具名与参数名必须完全匹配（区分大小写）。
        #     所有参数类型均为 Boolean；仅输出一个参数，并设为 True（其余不出现，视为 False）。
        #     只能选择一个工具；不得并列或降级为笼统意图。
        # 判定要点
        #     纯“查询/查看”语义 → 选该工具下的查询类参数（如 Check*、Search*、CheckBatteryLevel 等）。
        #     明确“执行/切换/开启/关闭/连接/发送/创建”等操作 → 选对应操作参数（如 SystemUpdate、BlueToothOnOff、ControlLuminance、CreateNote 等）。
        #     优先选择更专用的工具；仅当确为信息检索时才考虑通用搜索类工具。
        #     若用户已点名具体对象（如 Wi-Fi/蓝牙/飞行模式/系统更新/电源/回收站），优先选对应工具而非搜索。
        # 示例（示范格式，勿照抄内容）：
        #     "<tool>manageBluetooth(BlueToothOnOff=True)</tool>"
        #     "<tool>manageSystemUpdates(SystemUpdate=True)</tool>"
        # 工具列表：\n{self.tool_docs}
        # """

        # <<<   test use   <<<

        self.level_prompt = f"""
        你是一个小型工具路由器。

        任务：根据用户的自然语言指令，从给定的工具类别（function.name）及其下唯一一个 Boolean 类型子意图参数（properties里的键）中，选出最合适的一组（工具类别 + 子意图），并用指定格式输出。

        【工具类别说明】
        - 工具类别 = function.name，例如 placeCall、manageBluetooth、manageSystemUpdates 等。
        - 每个工具类别下都有若干 Boolean 子意图参数（properties），描述该工具的具体功能。
        - 你必须先选择**唯一的工具类别**，再从该工具下选择**唯一的子意图参数**。

        【选择规则】
        1. **选择提示与指导**
        - 请仔细理解用户意图，查看工具的描述，不要只根据工具名进行选择。
        - 仔细阅读函数下（即工具类别中）的可用参数的描述，如果用户意图符合参数的描述，选择这个工具类别。
        - 必须尽可能选择一个工具类别，不要直接执行指令（如翻译，搜索,查询任务）。
        - 对于多轮对话，请抓住要点，不要被疑问句迷惑，不要只看用户的最后一句，而要综合对话考虑。

        2. **匹配语义优先级**
        - 如果用户意图是纯查询或查看信息 → 选择该工具下的查询类参数（如 Check*、Search*、CheckBatteryLevel）。
        - 如果用户明确要执行/切换/开启/关闭/连接/发送/创建等操作 → 选择对应的执行类参数（如 Call、SystemUpdate、BlueToothOnOff）。
        - 当用户点名具体对象（如 Wi-Fi、蓝牙、飞行模式、系统更新、电源、回收站等），优先匹配其对应的专用工具，而不是通用搜索工具。

        3. **唯一性要求**
        - 工具类别和子意图参数必须完全匹配给定列表中的名称（区分大小写）。
        - 只能选择一个工具类别；只能输出它的一个 Boolean 子意图参数，并设为 True，其余参数视为 False。
        
        4. **输出格式**
        - 用 `<level></level>` 标签包裹类别调用代码。
        - 格式为：`<level>工具类别名(参数名=True)</level>`
        - 除这一行外不能输出任何额外的工具调用代码或说明文字。

        【禁止】
        - 不得选择多个工具类别或多个参数。
        - 不得输出笼统的意图描述或解释。
        - 不得修改工具类别名和参数名。

        【示范格式】（请勿照抄）
        - "<level>manageBluetooth(BlueToothOnOff=True)</level>"
        - "<level>manageSystemUpdates(SystemUpdate=True)</level>"

        【工具类别列表】
        {self.tool_docs}
        """

        mapping = {
        '设备电源操作': 'powerDevice',
        '电话操作': 'placeCall',
        '系统工具操作': 'runSystemUtility',
        '音量控制': 'setVolume',
        '字体设置': 'setFont',
        '搜索操作': 'performSearch',
        '主题与壁纸设置': 'setThemeAndWallpaper',
        '信息操作': 'manageMessages',
        '蓝牙管理': 'manageBluetooth',
        '系统更新管理': 'manageSystemUpdates',
        '语言与输入法设置': 'setLanguageAndInput',
        '省电模式设置': 'setPowerSavingMode',
        '显示参数调节': 'adjustDisplay',
        '电池状态查看': 'getBatteryStatus',
        '界面刷新': 'refreshUI',
        '飞行模式开关': 'toggleAirplaneMode',
        '网络控制': 'controlNetwork',
        '邮件操作': 'manageEmail',
        '回收站操作': 'manageRecycleBin',
        '应用切换': 'switchApp',
        '实际应用': 'launchApp',
        '闹钟操作': 'manageAlarms',
        '多任务视图': 'showTaskSwitcher',
        '声音模式切换': 'toggleSoundMode',
        }


        
        with open("prompts/tools_openai_format.json", "r", encoding="utf-8") as f:
            tools = json.load(f)
        level_tool = {}
        for tool in tools:
            category = mapping[tool['function']['parameters']['level'].strip()]
            if category not in level_tool.keys():
                level_tool[category] = []
            level_tool[category].append(str(tool))

        self.tool_system_prompt = {}
        for level,level_tool_list in level_tool.items():
            tools_info = "\n\n".join(level_tool_list)
            tool_prompt = f"""
            你是一个小型工具选择器。

            任务：根据用户的自然语言指令，从下方提供的**工具列表**中，选出最合适的一个工具，并按指定格式输出调用代码。

            【工具说明】
            1.工具列表中每个工具包含：
            - name：工具名称（函数名）
            - description：工具用途描述
            - parameters：该工具的入参（名字、类型、取值范围及说明）
            2.工具调用格式与 Python 函数一致：工具名(param1=value1, param2=value2, ...)

            【选择规则】
            1. 理解用户意图，匹配与 description 最贴近的工具。
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
            """

            self.tool_system_prompt[level] = tool_prompt


    def run(self, input_messages) -> str:
        # TODO: Implement your Agent logic here
        level_messages = [
                       {"role": "system", "content": self.level_prompt},
                   ] + input_messages
        response_content = self.generate(level_messages)
        levels = re.findall(r"<level>(.*?)</level>", response_content, re.DOTALL)
        if levels:
            level = levels[-1].strip()
            index = level.find('(')
            level = level[:index]
            if level in self.tool_system_prompt:
                tool_system_prompt = self.tool_system_prompt[level]
                tool_messages = [
                    {"role": "system", "content": tool_system_prompt},
                ] + input_messages
                response_content = self.generate(tool_messages)
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
