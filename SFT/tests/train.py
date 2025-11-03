# Test training of SFT with LoRA
# Kinda like the way without the noisy tab completions 
import json

from global_variables import level_prompt



# Data Preprocessing
class DataSet:
    def __init__(self,raw_data_path):
        self.raw_data=list()
        with open(raw_data_path,"r",encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.openai_x, self.openai_y= self._construct_openai(self.raw_data)
        

    def _construct_openai(self,raw_data:list):
        # container list
        res = []
        labels = []

        for item in raw_data:
            # example format: 
            # {
            #     "instruction": "帮我重启一下手机",
            #     "input": "",
            #     "output": "<level>powerDevice(Reboot=True)</level>"
            # }
            user_prompt = item["instruction"]
            answer = item["output"]
            # Currently it's single rounded. 
            message = []
            sys = {
                "role":"system",
                "content":level_prompt
            }
            usr = {
                "role":"user",
                "content": user_prompt
            }
            message += [sys,usr]
            res.append(message)
            labels.append(answer)

        return res,labels
    
    def get_openai_data(self):
        return self.openai_x
    
    def get_labels(self):
        return self.openai_y
    
    def get_openai_xy(self):
        return zip(self.openai_x,self.openai_y)
    
    

    


