import json

mode = input("Please input your checking mode(json/text):")

if mode == "json":
    json_path = "casual_tests/result_multiple.json"
    with open(json_path,"r",encoding='utf-8') as f:
        result = json.load(f)
    judge_path = "data/多轮-冒烟测试集.json"
    with open(judge_path,"r",encoding='utf-8') as f:
        corr_result = json.load(f)
    cnt = 0
    for i in range(len(result)):
        if result[i]['output'] != corr_result[i]['data'][-1]['content']:
            print(f"{i} -> res:{result[i]['output']},corr_res:{corr_result[i]['data'][-1]['content']}")
            cnt += 1
    print(cnt/len(result))

elif mode == "text":
    file_path = ""
    diff = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            index = 0
            if "->" in line:
                index = int(line.split("->", 1)[0].strip())
                line = line.split("->", 1)[1].strip()

            # 按 corr_res: 分割
            if ",corr_res:" in line:
                res_part, corr_part = line.split(",corr_res:", 1)
                res_val = res_part.replace("res:", "").strip()
                corr_val = corr_part.strip()
            else:
                print(f"第{i}行格式不正确: {line}")
                continue

            # 只输出不同的
            if res_val != corr_val:
                diff.append(index)
                print(f"第{i}行不一致:")
                print(f"res: {res_val}")
                print(f"corr_res: {corr_val}")
    print(diff)