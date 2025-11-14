from config import *
from data_model import *

import os
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

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


def main(cfg):
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
    dataset = ChatJsonlDataset(tok, cfg["data_path"], max_length=cfg["max_seq_length"],level=cfg["level"])
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
        report_to="tensorboard",
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
    cfg = CONFIG
    cfg = config_level(cfg)
    main(cfg)
    for level in level_list:
        cfg = config_leveled(cfg,level)
        main(cfg)
