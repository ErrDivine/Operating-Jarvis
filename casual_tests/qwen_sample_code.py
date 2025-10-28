from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/Users/errdivine/Desktop/tmp for Operating Jarvis/鸿蒙系统操控Agent竞赛/demo/models/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
#This is above api.
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]


#This is below api.
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)