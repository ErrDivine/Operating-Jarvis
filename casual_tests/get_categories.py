import json

level1_descriptions = {
    "placeCall": "拨打/接听电话并查询联系人",
    "runSystemUtility": "创建便签/备忘录",
    "setVolume": "调节媒体/通话/闹钟音量",
    "setFont": "查看并调整字体、字号与粗细",
    "performSearch": "系统内搜索内容或应用",
    "setThemeAndWallpaper": "切换深浅色或设置系统主题",
    "manageMessages": "发送短信或消息",
    "manageBluetooth": "开关蓝牙、搜索并连接设备",
    "manageSystemUpdates": "检查并下载/安装系统更新",
    "setLanguageAndInput": "切换系统语言与输入法",
    "setPowerSavingMode": "开启或配置省电模式",
    "adjustDisplay": "查询/调节亮度、对比度与自动亮度",
    "getBatteryStatus": "查询电量与耗电排行",
    "refreshUI": "刷新界面视图",
    "toggleAirplaneMode": "开启或关闭飞行模式",
    "controlNetwork": "开关WLAN/移动数据、搜索Wi-Fi、热点共享",
    "manageEmail": "发送或查看邮件",
    "powerDevice": "重启、关机或休眠设备",
    "manageRecycleBin": "清空回收站/废纸篓",
    "switchApp": "在前台应用之间切换",
    "launchApp": "翻译文本或截屏保存",
    "manageAlarms": "新建/查看/删除闹钟",
    "showTaskSwitcher": "打开任务管理器或切换多窗口",
    "toggleSoundMode": "切换振动与勿扰/静音模式",
}

# category list of level1 names
cat_list = ['placeCall', 'runSystemUtility', 'setVolume', 'setFont', 'performSearch',
                'setThemeAndWallpaper', 'manageMessages', 'manageBluetooth', 'manageSystemUpdates',
                'setLanguageAndInput', 'setPowerSavingMode', 'adjustDisplay', 'getBatteryStatus',
                'refreshUI', 'toggleAirplaneMode', 'controlNetwork', 'manageEmail', 'powerDevice',
                'manageRecycleBin', 'switchApp', 'launchApp', 'manageAlarms', 'showTaskSwitcher',
                'toggleSoundMode']
mapping = {
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
        '设备电源操作': 'powerDevice',
        '回收站操作': 'manageRecycleBin',
        '应用切换': 'switchApp',
        '实际应用': 'launchApp',
        '闹钟操作': 'manageAlarms',
        '多任务视图': 'showTaskSwitcher',
        '声音模式切换': 'toggleSoundMode',
    }


def read_tool_json(input_file="../prompts/tools_v0.json"):
    with open(input_file, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    return tools_data


def _get_categories(input_file):
    tools_data = read_tool_json(input_file)

    categories = set()
    for tool in tools_data:
        categories.add(tool["level1"])

    return categories

def map_to_list(map):
    res = []
    for key, value in map.items():
        res.append(value)

    return res

def get_categoried_tools():
    tools_jsons = read_tool_json()

    res = dict()

    # Initialization
    for cat in cat_list:
        res[cat] = []

    # Fill
    for tool in tools_jsons:
        # English name of the tool's level1 name
        level1 = mapping[tool["level1"]]

        # add tools in the list 
        res[level1].append(tool["name"])

    # print(res)
    return res


def construct_cat_openai_json():

    tools_jsons = read_tool_json()
    res = []
    for cat in cat_list:
        chinese_cat = ''
        for key, value in mapping.items():
            if value == cat:
                chinese_cat = key
                break

        desc = level1_descriptions[cat]
        function_def = {
            "name": cat,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        # Iteratively find all corresponding tools and add names and descs into the properties.
        for tool in tools_jsons:
            if tool["level1"] == chinese_cat:
                name = tool["name"]
                function_def["parameters"]["properties"][name] ={"type": "Boolean",
                                                                 "description": tool["description"]
                                                                 }

        res.append({
            "type":"function",
            "function": function_def,
        })


    return res




if __name__ == '__main__':

    # print(len(read_tool_json()))
    # list = map_to_list(mapping)
    # print(len(list))

    cat_tools = get_categoried_tools()
    cnt = 0
    for key, value in cat_tools.items():
        cnt += len(value)
        print(key, ":",value)

    # cons = construct_cat_openai_json()
    # with open("test_level1_json.json", "w", encoding="utf-8") as f:
    #     json.dump(cons, f,ensure_ascii=False, indent=2)

    # print(cons)