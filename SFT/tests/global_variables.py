import json
level1_json_path = "models/test_level1_json.json"
# >>>   level1 use   >>>
with open(level1_json_path, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
tool_docs = str(tools_data)






level_prompt = f"""
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
        {tool_docs}
        """