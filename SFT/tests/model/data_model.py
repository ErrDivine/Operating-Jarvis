# Data model. Preprocessing of training data.
import json
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self,data_path,tokenizer):
        # raw data
        raw_json = []
        with open("single.json","r",encoding='utf-8') as f:
            self.raw_json = json.load(f)
