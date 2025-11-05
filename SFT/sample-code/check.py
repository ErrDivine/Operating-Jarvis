import json
from tqdm import tqdm
import re 

from infer import chat_once, load_lora_chat_model, SystemPrompts

def get_validation_msgs(path):
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # load model
    direct_sys = SystemPrompts().direct
    tok, model = load_lora_chat_model(
        base_model="../../models/Qwen3-4B-Instruct-2507",
        adapter_dir="direct-lora",
    )
    msgs = get_validation_msgs("/root/autodl-tmp/Operating-Jarvis/SFT/data/single.json")
    with open("check_result.txt","w",encoding="utf-8") as f:
        for msg in tqdm(msgs[:50]):
            msg = [{"role":"system","content":direct_sys}]+msg["input"]
            rret = chat_once(tok,model,msg[:-1])
            try:
                ret = re.findall(r"assistant\n(<tool>.+</tool>)",rret)[0].strip()
            except:
                ret = rret
            # print(ret)
            answer = msg[-1]["content"]
            if ret != answer:
                # conversation = re.findall(r"user(.+</tool>)",rret)[0].strip()
                f.write(f"\n{"*"*50}\nagent:{rret}\nanswer:{answer}\n")


