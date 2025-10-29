#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2-4B-Instruct"
OUT="out-qwen4b-lora"

python train_lora_qwen.py \  --model_name_or_path "$MODEL" \  --data_path data/sample.jsonl \  --output_dir "$OUT" \  --num_train_epochs 1 \  --per_device_train_batch_size 1 \  --gradient_accumulation_steps 8 \  --learning_rate 2e-4 \  --bf16 \  --use_qlora \  --gradient_checkpointing \  --max_seq_length 2048

python infer_chat.py --model_name_or_path "$MODEL" --adapter_dir "$OUT" --max_new_tokens 96
