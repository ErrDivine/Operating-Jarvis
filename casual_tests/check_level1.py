import re
import json

from get_categories import get_categoried_tools


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


def _compare_categories(res):  # don't print if everything is correct
    map = get_categoried_tools()
    for item in res:
        answer = item["input"]["data"][-1]["content"]
        answer_pattern = r'\w+\('
        try:
            answer = re.search(answer_pattern, answer).group()[:-1]
        except:
            print(answer)
            continue
        ret = item["output"]
        # print(ret)
        ret_pattern = r'\w+\('
        try:
            ret = re.search(ret_pattern, ret).group()[:-1]
        except:
            print(answer, ret)

        try:
            if answer not in map[ret]:  # if the category is wrong, go into this branch
                for key, value in map.items():
                    if answer in value:
                        answer_cat = key
                        break
                print(answer, answer_cat, ret, map[ret], '\n\n')
                '''
                answer: correct tool name
                answer_cat: correct category name
                ret: model output category name
                map[ret]: tools to be chosen of a category
                '''
        except:
            print(ret, '\n')


def compare_categories():
    res_single = load_data("result_category_single.json")
    res_multi = load_data("result_category_multiple.json")

    _compare_categories(res_single)
    _compare_categories(res_multi)


if __name__ == '__main__':
    compare_categories()

