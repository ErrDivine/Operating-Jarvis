# Check qwen correctness
import re


from train import DataSet
from agent import CustomAgent

# Monkey patching the run method in agent
def level_run(self,level_messages):
        response_content = self.generate(level_messages)
        level_res = re.findall(r".*?(<level>.*?</level>).*?", response_content, re.DOTALL)
        return level_res


def main():
    CustomAgent.run = level_run
    agent = CustomAgent()
    data_set = DataSet("SFT/tests/tool_classification_lora_dataset.json")
    for message, answer in data_set.get_openai_xy():
        response = agent.run(message)
        if response != answer :
            print('*'*20,f"\nres:{response}\ncor_res{answer}\n")



if __name__ == "__main__":
     main()

        
