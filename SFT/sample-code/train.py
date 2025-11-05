#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hardcoded-params LoRA (QLoRA) SFT for Qwen-* Instruct on OpenAI-style chat JSONL.
Supervises **only the last assistant turn** per sample.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
from infer import SystemPrompts

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ============== HARD-CODED CONFIG ==============
CONFIG = {
    # Base model & data
    "model_name_or_path": "../../models/Qwen3-4B-Instruct-2507",
    "data_path": "../data/single.json",
    "output_dir": "out-qwen4b-lora-lastonly",

    # Repro
    "seed": 42,

    # Training schedule
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,

    # Lengths
    "max_seq_length": 2048,

    # Precision (set bf16 True if available on your GPU)
    "bf16": True,
    "fp16": False,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Qwen/LLaMA-like target modules:
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",

    # Efficiency
    "gradient_checkpointing": True,
    "save_steps": 0,
    "save_strategy": "epoch",
    "logging_steps": 10,

    # Quantization (QLoRA)
    "use_qlora": True,
    "bnb_4bit_compute_dtype": "bfloat16",  # bfloat16|float16|float32
}
# ===============================================


def make_bnb_config(cfg):
    if not cfg["use_qlora"]:
        return None
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg["bnb_4bit_compute_dtype"]]

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def role_is_assistant(m):
    return m.get("role") == "assistant"


def get_encoder_and_tokenspan(tokenizer, messages: List[Dict[str, str]]) -> Tuple[dict, List[Tuple[int, int]]]:
    """
    Render full conversation via chat template; return ONLY the span of the *last*
    assistant segment (if any). Otherwise return empty span list.
    """
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    ) or ""
    before_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=False
    )
    c_start=len(before_text)
    c_end=len(full_text)
    # print(c_start,c_end)
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"][0].tolist() # list[(s,e)]
    # print(offsets)
    tok_start = None
    tok_end = None
    for ti, (ts, te) in enumerate(offsets):
        if te <= c_start:
            continue
        if ts >= c_end:
            break
        if tok_start is None:
            tok_start = ti
        tok_end = ti + 1
    if tok_start is not None and tok_end is not None:
        token_span=(tok_start, tok_end)
    # print(tok_start,tok_end)
    return enc, token_span


def build_labels(input_ids: torch.Tensor, token_span: Tuple[int, int]):
    labels = torch.full_like(input_ids, -100)
    s, e = token_span
    labels[0, s:e] = input_ids[0, s:e]
    return labels


class ChatJsonlDataset(Dataset):
    def __init__(self, tokenizer, path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for element in data:
                self.samples.append(element)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]["input"]
        direct_sys = SystemPrompts().direct
        messages = [direct_sys] + messages
        # print(messages)
        enc, token_spans = get_encoder_and_tokenspan(self.tokenizer, messages)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = build_labels(input_ids, token_spans)

        # Left-truncate to keep recent context
        if input_ids.size(1) > self.max_length:
            cut = input_ids.size(1) - self.max_length
            input_ids = input_ids[:, cut:]
            attn = attn[:, cut:]
            labels = labels[:, cut:]

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attn.squeeze(0),
            "labels": labels.squeeze(0),
        }


@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features):
        b_input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id
        )
        b_attn = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        b_labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        )
        return {"input_ids": b_input_ids, "attention_mask": b_attn, "labels": b_labels}


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # Quantization config if QLoRA
    bnb_config = make_bnb_config(cfg)

    print(f"[Config] {cfg}")
    print(f"Loading tokenizer: {cfg['model_name_or_path']}")
    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not os.path.exists(cfg["data_path"]):
        raise FileNotFoundError(f"Data file not found: {cfg['data_path']} (cwd={os.getcwd()})")

    print(f"Loading base model: {cfg['model_name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=(torch.bfloat16 if cfg["bf16"] else (torch.float16 if cfg["fp16"] else None)),
        device_map="auto",
    )

    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    if cfg["use_qlora"]:
        print("Preparing model for k-bit training (QLoRA) ...")
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in cfg["target_modules"].split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Loading dataset ...")
    dataset = ChatJsonlDataset(tok, cfg["data_path"], max_length=cfg["max_seq_length"])
    collator = DataCollator(pad_token_id=tok.pad_token_id)

    print("Setting up Trainer ...")
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_strategy=cfg["save_strategy"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        optim=("paged_adamw_8bit" if cfg["use_qlora"] else "adamw_torch"),
        report_to="none",
        max_grad_norm=cfg["max_grad_norm"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Start training (last-assistant-only, hardcoded params) ...")
    trainer.train()

    print("Saving adapter ...")
    trainer.model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

    print("All done. Inference tip:")
    print(f"  python infer_chat.py --model_name_or_path \"{cfg['model_name_or_path']}\" --adapter_dir \"{cfg['output_dir']}\" --max_new_tokens 128")


if __name__ == "__main__":
    main()
