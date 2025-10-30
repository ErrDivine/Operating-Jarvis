import json
from tqdm import tqdm

from agent import CustomAgent
from demo_agent import DirectAgent, HierarchicalAgent
from teacher import call_teacher

def pre_input(json_data):
    # input = []
    # for turn in json_data["data"]:
    #     if turn["role"] == "assistant" and turn["content"][1].isalpha():
    #         break
    #     input.append(turn)
    # return input
    turn = json_data['data']
    return turn[:-1]


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


if __name__ == "__main__":
    # data = load_data("data/单轮-冒烟测试集.jsonl")
    data = load_data("data/多轮-冒烟测试集.jsonl")
    # agent = DirectAgent()
    # agent = HierarchicalAgent()
    agent = CustomAgent("models/Qwen3-4B-Instruct-2507")

    results = []
    for item in tqdm(data):
        response = agent.run(pre_input(item))
        # response = call_teacher(pre_input(item))
        results.append({"input": pre_input(item), "output": response})
        # print("Input:", item)
        # print("Output:", response)
        # print("-" * 50)
    with open("casual_tests/result_category_multiple.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
