import json

def jsonl_to_json(jsonl_path, json_path):
    """
    将 .jsonl 文件转换为 .json（数组）文件
    :param jsonl_path: 输入的 jsonl 文件路径
    :param json_path: 输出的 json 文件路径
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"解析失败: {e}，原始行: {line}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"转换完成：{json_path}（共 {len(data)} 条记录）")

if __name__ == "__main__":
    flag = input("Do you need to input path(T/F):")
    if flag == "T":
        jsonl_path = input("Please input jsonl path:")
        json_path = input("Please input json path:")
    else:
        jsonl_path = "data/单轮-冒烟测试集.jsonl"
        json_path = "data/单轮-冒烟测试集.json"
    jsonl_to_json(jsonl_path, json_path)
