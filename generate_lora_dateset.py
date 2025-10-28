from google import genai
from google.genai import types
import json
import time

# -------------------------- 1. 基础配置 --------------------------
# Gemini API配置
API_KEY = "sk-USul5hvMppKKq7updrxOCPsR7MGHXuDsGa1JPYJbCZbOVVu5"  # 替换为你的Gemini API密钥
BASE_URL = "https://turingai.plus"  # API端点

# 工具类别映射
mapping = {
    '设备电源操作': 'powerDevice', '电话操作': 'placeCall', '系统工具操作': 'runSystemUtility',
    '音量控制': 'setVolume', '字体设置': 'setFont', '搜索操作': 'performSearch',
    '主题与壁纸设置': 'setThemeAndWallpaper', '信息操作': 'manageMessages',
    '蓝牙管理': 'manageBluetooth', '系统更新管理': 'manageSystemUpdates',
    '语言与输入法设置': 'setLanguageAndInput', '省电模式设置': 'setPowerSavingMode',
    '显示参数调节': 'adjustDisplay', '电池状态查看': 'getBatteryStatus',
    '界面刷新': 'refreshUI', '飞行模式开关': 'toggleAirplaneMode',
    '网络控制': 'controlNetwork', '邮件操作': 'manageEmail',
    '回收站操作': 'manageRecycleBin', '应用切换': 'switchApp',
    '实际应用': 'launchApp', '闹钟操作': 'manageAlarms',
    '多任务视图': 'showTaskSwitcher', '声音模式切换': 'toggleSoundMode'
}

# 输出文件路径
output_file = "tool_classification_lora_dataset.json"
# 批量生成参数
batch_size = 10  # 每批生成数量
total_batches = 5  # 总批次数
all_samples = []


# -------------------------- 2. Gemini API调用函数 --------------------------
def generate_batch_samples(batch_num, batch_size, mapping):
    """使用Gemini API生成一批样本"""

    # 创建Gemini客户端 - 修复http_options格式
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(base_url=BASE_URL)  # 修正这里
    )

    category_list = list(mapping.values())

    prompt = f"""
你是一个工具分类数据生成器。需要为工具分类模型生成训练数据。

## 任务要求：
生成{batch_size}条用户指令与对应工具分类的样本数据。

## 工具类别系统：
可用的工具类别：{category_list}

## 输出格式要求：
返回一个JSON数组，每个元素是包含三个字段的对象：
{{
    "instruction": "用户自然语言指令",
    "input": "", 
    "output": "<level>工具类别名(参数名=True)</level>"
}}

## 生成规则：
1. 指令多样性：涵盖日常手机操作场景（调节设置、查询状态、执行操作等）
2. 类别分布：尽量均匀使用所有工具类别
3. 参数选择：为每个工具类别选择合适的Boolean参数
4. 指令自然：使用真实用户会说的自然语言
5. 格式严格：output必须严格遵循 `<level>类别名(参数名=True)</level>` 格式

## 示例：
[
    {{
        "instruction": "帮我打开蓝牙",
        "input": "",
        "output": "<level>manageBluetooth(BlueToothOnOff=True)</level>"
    }},
    {{
        "instruction": "查看手机电量还有多少",
        "input": "", 
        "output": "<level>getBatteryStatus(CheckBatteryLevel=True)</level>"
    }},
    {{
        "instruction": "关闭WiFi连接",
        "input": "",
        "output": "<level>controlNetwork(WifiOnOff=True)</level>"
    }}
]

请直接返回JSON数组，不要有其他文字说明。
"""

    try:
        # 调用Gemini API
        response = client.models.generate_content(
            model='models/gemini-2.5-pro',
            contents=types.Content(
                parts=[
                    types.Part(text=prompt)
                ]
            )
        )

        # 提取响应文本
        response_text = response.text.strip()
        print(f"第{batch_num}批原始响应: {response_text[:200]}...")  # 打印前200字符用于调试

        # 尝试解析JSON响应
        try:
            # 清理响应文本，确保是有效的JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            batch_samples = json.loads(response_text)

            # 验证样本格式
            valid_samples = []
            for sample in batch_samples:
                if (isinstance(sample, dict) and
                        "instruction" in sample and
                        "input" in sample and
                        "output" in sample and
                        sample["output"].startswith("<level>") and
                        sample["output"].endswith("</level>")):
                    valid_samples.append(sample)
                else:
                    print(f"跳过无效样本格式：{sample}")

            return valid_samples

        except json.JSONDecodeError as e:
            print(f"JSON解析失败（第{batch_num}批）：{e}")
            print(f"原始响应：{response_text}")
            return []

    except Exception as e:
        print(f"API调用失败（第{batch_num}批）：{str(e)}")
        time.sleep(5)  # 失败后重试等待
        return []


# -------------------------- 3. 批量生成与保存 --------------------------
def main():
    """主函数：批量生成测试集"""
    print("开始生成工具分类测试集...")

    # 多轮调用API生成样本
    for batch in range(1, total_batches + 1):
        print(f"正在生成第{batch}批样本...")
        batch_samples = generate_batch_samples(batch, batch_size, mapping)
        if batch_samples:
            all_samples.extend(batch_samples)
            print(f"第{batch}批生成成功，获得{len(batch_samples)}条样本，累计样本数：{len(all_samples)}")
        else:
            print(f"第{batch}批生成失败")

        time.sleep(2)  # 避免API请求频率超限

    # 保存所有样本到JSON文件
    if all_samples:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"批量生成完成！共生成{len(all_samples)}条样本，已保存到 {output_file}")

        # 统计类别分布
        category_count = {}
        for sample in all_samples:
            # 提取工具类别名
            output = sample["output"]
            start = output.find("<level>") + 7
            end = output.find("(")
            if start != -1 and end != -1:
                category = output[start:end]
                category_count[category] = category_count.get(category, 0) + 1

        print("\n类别分布统计：")
        for category, count in sorted(category_count.items()):
            print(f"  {category}: {count}条")

        print(f"\n总样本数: {len(all_samples)}")
    else:
        print("未生成任何有效样本")


if __name__ == "__main__":
    main()