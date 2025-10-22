import json
import re

def load_data(data_path):
    if data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    else:
        raise ValueError("Only json and jsonl files are supported")

def compare_categories():
    res_single = load_data("../results_single.json")
    res_multi = load_data("../results_multiple.json")

    _compare_categories(res_single)
    _compare_categories(res_multi)


def _compare_categories(res):
    for item in res:
        answer = item["input"]["data"][-1]["content"]
        answer_pattern = r'\w+('
        answer = re.search(answer_pattern, answer).group()[:-1]

        ret = item["output"]
        ret_pattern = r'(\w+)'
        ret = re.search(ret_pattern, ret).group()[1:-1]

        if answer != ret:
            print('*'*50)
            print(answer)
            print(ret,'\n\n')


json_path = "../results/results_single.json"
json_path_2 = "../results/results_multiple.json"
result=load_data(json_path)
result_2=load_data(json_path_2)
judge_path = "../data/单轮-冒烟测试集.jsonl"
corr_result = load_data(judge_path)
judge_path_2="../data/多轮-冒烟测试集.jsonl"
corr_result_2 = load_data(judge_path_2)

def check(result,bench):
    num = 0
    with open("test_result.txt", "a", encoding="utf-8") as f:
        for i in range(len(result)):
            term = result[i]
            benchmark = bench[i]
            res = term['output']
            ans = benchmark["data"][-1]["content"]
            if res != ans:
                print(f"{res} != {ans}\n")
                f.write(f"{'*'*50}\n\nAgent output:{res}\nCorrect answer:{ans}{'*'*50}")
                num += 1
    return num

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("restart")
stat = 0
stat += check(result,corr_result)
stat += check(result_2,corr_result_2)
print(f"Total:{len(result)+len(result_2)},mismatched:{stat}")


