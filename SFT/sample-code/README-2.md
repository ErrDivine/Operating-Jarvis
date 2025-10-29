# Qwen-4B LoRA SFT (OpenAI-style chats)

This is a minimal, reproducible example to fine-tune a Qwen-* Instruct model with LoRA (QLoRA by default) using chat data stored as **OpenAI-style** `messages` lists. It mirrors the workflow in your `LLM-Finetuning-Maintained/1.Efficiently_train_Large_Language_Models_with_LoRA_and_Hugging_Face.ipynb`, but adapted to **Qwen chat templates** and **assistant-only loss**.

## Files
- `train_lora_qwen.py` — training script (LoRA + optional QLoRA).
- `infer_chat.py` — quick smoke test for the trained adapter.
- `data/sample.jsonl` — tiny sample dataset (OpenAI-style messages).
- `requirements.txt` — pinned versions.
- `run.sh` — example command lines.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (China) optional mirrors:
# export HF_ENDPOINT=https://hf-mirror.com
```

> **CUDA note**: match your `torch` to the CUDA in your driver. If needed, install a different `torch` wheel from https://pytorch.org/get-started/locally/ and keep the rest as-is.

## Data format
Each line is a JSON object with a `messages` list. Roles: `system` (optional), `user`, `assistant`. The trainer renders each conversation with `tokenizer.apply_chat_template` and **masks loss to assistant segments only**.

```json
{"messages":[
  {"role":"system","content":"You are a helpful, concise assistant."},
  {"role":"user","content":"What is a red–black tree?"},
  {"role":"assistant","content":"A red–black tree is a balanced BST with ..."},
  {"role":"user","content":"Why does insertion stay O(log n)?"},
  {"role":"assistant","content":"Because rebalancing touches a constant number of nodes ..."}
]}
```

## Run (QLoRA, single 4B model, bf16 compute)
Set your exact model id (examples: `Qwen/Qwen2-4B-Instruct`, `Qwen/Qwen1.5-4B-Chat`).

```bash
# train
python train_lora_qwen.py   --model_name_or_path "Qwen/Qwen2-4B-Instruct"   --data_path data/sample.jsonl   --output_dir out-qwen4b-lora   --num_train_epochs 1   --per_device_train_batch_size 1   --gradient_accumulation_steps 8   --learning_rate 2e-4   --bf16   --use_qlora   --gradient_checkpointing   --max_seq_length 2048

# infer (smoke test)
python infer_chat.py   --model_name_or_path "Qwen/Qwen2-4B-Instruct"   --adapter_dir out-qwen4b-lora   --max_new_tokens 128
```

### Tips
- If you see a deprecation warning like “use BitsAndBytesConfig via quantization_config”, we already do that.
- If tokenizer has no pad token, the script reuses EOS as PAD.
- To change LoRA targets, edit `--target_modules` (defaults for LLaMA-like Qwen blocks: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`).

## Why assistant-only masking?
We want the model to **condition** on system/user text and **predict** assistant text. The dataset collator computes token spans for each assistant turn and sets all other labels to `-100` so they don't contribute to loss.

## Expected VRAM
With QLoRA + gradient checkpointing and batch size 1, a 4B Instruct model is comfortable on a single 24GB GPU; 16GB may also work with smaller `max_seq_length` and more accumulation.

## Repro/seed
Default seed is 42; set `--seed` to fix randomness across runs.
