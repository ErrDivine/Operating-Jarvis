CONFIG = {
    # Base model & data
    "model_name_or_path": "../../models/Qwen3-4B-Instruct-2507",
    "data_path": "../data/single.json",
    "output_dir": "direct-lora",

    # Repro
    "seed": 1221,

    # Training schedule
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.3,

    # Lengths
    "max_seq_length": 4096,

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
    "gradient_checkpointing": False,
    "save_steps": 0,
    "save_strategy": "no",
    "logging_steps": 50,

    # Quantization (QLoRA)
    "use_qlora": True,
    "bnb_4bit_compute_dtype": "float16",  # bfloat16|float16|float32
}

def config_level(cfg):
    cfg["output_dir"]="lora_adapters/level"
    cfg["data_path"]="data/level.json"
    cfg["output_dir"]="lora_adapters/level"
    cfg["level"] = "level"
    return cfg

def config_leveled(cfg,level):
    cfg["output_dir"]=f"lora_adapters/{level}"
    cfg["data_path"]=f"data/{level}.json"
    cfg["output_dir"]=f"lora_adapters/{level}"
    cfg["level"] = level
    return cfg

