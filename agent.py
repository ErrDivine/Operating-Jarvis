import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer

IS_NPU = False
try:
    import torch_npu

    if torch_npu.npu.is_available():
        IS_NPU = True
    else:
        IS_NPU = False
except ImportError:
    IS_NPU = False


class BaseLLM(ABC):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto" if not IS_NPU else {"": "npu"},
        )

    def generate(self, messages, max_new_tokens=1024, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, **kwargs
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content


class CustomAgent:
    def __init__(self):
        # TODO: Initialize your Agent here
        raise NotImplementedError("CustomAgent is not implemented yet.")

    def run(self, input_messages) -> str:
        # TODO: Implement your Agent logic here
        raise NotImplementedError("CustomAgent run method is not implemented yet.")
        return "[Your Agent response]"
