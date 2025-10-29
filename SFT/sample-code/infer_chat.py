#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick smoke test for the trained LoRA adapter.
"""


import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Base model, e.g., Qwen/Qwen2-4B-Instruct")
    p.add_argument("--adapter_dir", type=str, required=True,
                   help="LoRA output dir from training.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Explain binary search in one concise paragraph."},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                         max_new_tokens=args.max_new_tokens)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
