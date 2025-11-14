import torch
from torch.utils.data import Dataset
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
level_list = [
    "manageBluetooth",
    "controlNetwork",
    "getBatteryStatus",
    "setPowerSavingMode",
    "powerDevice",
    "setVolume",
    "toggleSoundMode",
    "adjustDisplay",
    "setThemeAndWallpaper",
    "setFont",
    "setLanguageAndInput",
    "toggleAirplaneMode",
    "manageSystemUpdates",
    "switchApp",
    "placeCall",
    "manageMessages",
    "manageEmail",
    "manageAlarms",
    "runSystemUtility",
    "launchApp",
    "refreshUI",
    "showTaskSwitcher",
    "manageRecycleBin",
    "performSearch",
]


class SystemPrompts:
    def __init__(self):
        self.level1_json_path = "../models/test_level1_json.json"
        # >>>   level1 use   >>>
        with open(self.level1_json_path, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
        self.tool_docs = str(tools_data)
        self.direct = self._get_direct()
        self.level = self._get_level()
        self.tools_by_level = {
            "manageBluetooth": [
                "BlueToothOnOff",
                "SearchBlueTooth",
                "ConnectBlueTooth",
            ],
            "controlNetwork": [
                "WlanOnOff",
                "SearchWlan",
                "MobileDataOnOff",
                "HotShotOnOff",
            ],
            "getBatteryStatus": [
                "CheckBatteryLevel",
                "CheckBatteryConsumptionRank",
            ],
            "setPowerSavingMode": [
                "BatterySavingMode",
            ],
            "powerDevice": [
                "Reboot",
                "ShutDown",
                "Sleep",
            ],
            "setVolume": [
                "ControlSound",
            ],
            "toggleSoundMode": [
                "VibrationModeOnOff",
                "QuietModeOnOff",
            ],
            "adjustDisplay": [
                "CheckLuminance",
                "ControlLuminance",
                "AutoLuminanceOnOff",
                "CheckContrast",
                "SwitchContrast",
            ],
            "setThemeAndWallpaper": [
                "SetSystemTheme",
            ],
            "setFont": [
                "CheckFont",
                "SwitchFontSize",
                "SwitchFontWeight",
                "ControlFontType",
            ],
            "setLanguageAndInput": [
                "CheckInputMethod",
                "SwitchInputMethod",
                "SwitchLanguage",
            ],
            "toggleAirplaneMode": [
                "AirplaneModeOnOff",
            ],
            "manageSystemUpdates": [
                "CheckSystemUpdate",
                "SystemUpdate",
            ],
            "switchApp": [
                "SwitchApp",
            ],
            "placeCall": [
                "Call",
                "CheckContact",
                "AnswerCall",
            ],
            "manageMessages": [
                "SendMessage",
            ],
            "manageEmail": [
                "SendEmail",
                "CheckEmail",
            ],
            "manageAlarms": [
                "CreateAlarm",
                "CheckAlarm",
                "DeleteAlarm",
            ],
            "runSystemUtility": [
                "CreateNote",
            ],
            "launchApp": [
                "Translate",
                "CaptureScreenshot",
            ],
            "refreshUI": [
                "Refresh",
            ],
            "showTaskSwitcher": [
                "TaskManagerOnOff",
                "MultipleWindowModeOnOff",
            ],
            "manageRecycleBin": [
                "EmptyBin",
            ],
            "performSearch": [
                "Search",
            ],
        }

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
        
        self.tools = {}

        # extract leveled tool info list
        tools_json_path = "../models/tools_openai_format.json"
        with open(tools_json_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        level_tool = {}
        for tool in tools:
            category = mapping[tool['function']['parameters']['level'].strip()]
            if category not in level_tool.keys():
                level_tool[category] = []
            level_tool[category].append(str(tool))

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

            self.tools[level] = tool_prompt
        


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
    
    def _get_level(self):
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
        {self.tool_docs}
        """

        return level_prompt
    


class ChatJsonlDataset(Dataset):
    def __init__(self, tokenizer, path: str, max_length: int, level: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.level = level

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for element in data:
                self.samples.append(element)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]["message"]
        if self.level == "level":
            system_prompt = {"role":"system","content":SystemPrompts().level}
        else:
            system_prompt = {"role":"system","content":SystemPrompts().tools[self.level]}
        messages = [system_prompt] + messages

        # print(messages)
        enc, token_spans = self.get_encoder_and_tokenspan(self.tokenizer, messages)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = self.build_labels(input_ids, token_spans)

        # Left-truncate to keep recent context
        if input_ids.size(1) > self.max_length:
            cut = input_ids.size(1) - self.max_length
            input_ids = input_ids[:, cut:]
            attn = attn[:, cut:]
            labels = labels[:, cut:]

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attn.squeeze(0),
            "labels": labels.squeeze(0),
        }
    
    def get_encoder_and_tokenspan(self,tokenizer, messages: List[Dict[str, str]]) -> Tuple[dict, List[Tuple[int, int]]]:
        """
        Render full conversation via chat template; return ONLY the span of the *last*
        assistant segment (if any). Otherwise return empty span list.
        """
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ) or ""
        before_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=False
        )
        c_start=len(before_text)
        c_end=len(full_text)
        # print(c_start,c_end)
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"][0].tolist() # list[(s,e)]
        # print(offsets)
        tok_start = None
        tok_end = None
        for ti, (ts, te) in enumerate(offsets):
            if te <= c_start:
                continue
            if ts >= c_end:
                break
            if tok_start is None:
                tok_start = ti
            tok_end = ti + 1
        if tok_start is not None and tok_end is not None:
            token_span=(tok_start, tok_end)
        # print(tok_start,tok_end)
        return enc, token_span


    def build_labels(self,input_ids: torch.Tensor, token_span: Tuple[int, int]):
        labels = torch.full_like(input_ids, -100)
        s, e = token_span
        labels[0, s:e] = input_ids[0, s:e]
        return labels


@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features):
        b_input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id
        )
        b_attn = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        b_labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        )
        return {"input_ids": b_input_ids, "attention_mask": b_attn, "labels": b_labels}