#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA SFT for Qwen-* Instruct models on OpenAI-style chat data.

- Trains only on assistant tokens (user/system/tool masked out).
- Uses QLoRA (4-bit) by default to fit on a single 24GB+ GPU for 4B models.
- Render messages via tokenizer.apply_chat_template to stay consistent with Qwen chat format.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="e.g., Qwen/Qwen2-4B-Instruct (set to your exact checkpoint id)")
    p.add_argument("--data_path", type=str, required=True,
                   help="JSONL file with {\"messages\": [...] } per line in OpenAI style.")
    p.add_argument("--output_dir", type=str, default="qwen4b-lora-out")
    p.add_argument("--seed", type=int, default=42)

    # training
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # lengths
    p.add_argument("--max_seq_length", type=int, default=3072)
    p.add_argument("--packing", action="store_true",
                   help="Pack multiple samples into one sequence (simple naive pack).")

    # precision
    p.add_argument("--bf16", action="store_true", help="Use bfloat16")
    p.add_argument("--fp16", action="store_true", help="Use float16")

    # lora
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated module names to apply LoRA to.")

    # deepspeed/accelerate-like knobs
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--logging_steps", type=int, default=10)

    # quantization
    p.add_argument("--use_qlora", action="store_true", help="Enable 4-bit QLoRA.")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    return p.parse_args()


def make_bnb_config(args):
    if not args.use_qlora:
        return None
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.bnb_4bit_compute_dtype]

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def role_is_assistant(m):
    return m.get("role") == "assistant"


def find_assistant_char_spans(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Render the conversation with the chat template and return:
      - full rendered text
      - a list of (char_start, char_end) spans that correspond to assistant *segments*
    We compute spans by incremental rendering: text_before vs text_with_current.
    We include the entire assistant segment produced by the template (including any role markers).
    """
    rendered_spans = []
    text_before = tokenizer.apply_chat_template(
        [], tokenize=False, add_generation_prompt=False
    ) or ""

    full_text = text_before
    for i, _ in enumerate(messages):
        text_up_to_i = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        delta = text_up_to_i[len(full_text):]
        if role_is_assistant(messages[i]):
            start = len(full_text)
            end = len(text_up_to_i)
            rendered_spans.append((start, end))
        full_text = text_up_to_i

    return full_text, rendered_spans


def char_spans_to_token_spans(tokenizer, text: str, char_spans: List[Tuple[int, int]]):
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"][0].tolist()  # list of (s, e)
    token_spans = []
    for (cs, ce) in char_spans:
        # collect tokens whose offsets intersect [cs, ce)
        tok_start = None
        tok_end = None
        for ti, (ts, te) in enumerate(offsets):
            if te <= cs:
                continue
            if ts >= ce:
                break
            if tok_start is None:
                tok_start = ti
            tok_end = ti + 1
        if tok_start is not None and tok_end is not None:
            token_spans.append((tok_start, tok_end))
    return enc, token_spans


def build_labels(input_ids: torch.Tensor, token_spans: List[Tuple[int, int]]):
    labels = torch.full_like(input_ids, -100)
    for s, e in token_spans:
        labels[0, s:e] = input_ids[0, s:e]
    return labels


class ChatJsonlDataset(Dataset):
    def __init__(self, tokenizer, path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                assert "messages" in obj, "Each line must have a `messages` list."
                self.samples.append(obj["messages"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        text, char_spans = find_assistant_char_spans(self.tokenizer, messages)
        enc, token_spans = char_spans_to_token_spans(self.tokenizer, text, char_spans)

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = build_labels(input_ids, token_spans)

        # truncate from left if too long (keep recent context)
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

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # simple left-pad = False (right pad) collator
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
    args = parse_args()
    set_seed(args.seed)

    bnb_config = make_bnb_config(args)

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        # Qwen chat models usually have a pad token defined; if not, fall back to eos
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        print("Preparing model for k-bit training (QLoRA) ...")
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset ...")
    dataset = ChatJsonlDataset(tokenizer, args.data_path, max_length=args.max_seq_length)
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)

    print("Setting up Trainer ...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        report_to="none",
        max_grad_norm=args.max_grad_norm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Start training ...")
    trainer.train()

    print("Saving adapter ...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("All done. Try inference with infer_chat.py --adapter_dir", args.output_dir)


if __name__ == "__main__":
    main()
