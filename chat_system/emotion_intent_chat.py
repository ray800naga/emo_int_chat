import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

config  = PeftConfig.from_pretrained("/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20240908_135426_Joy")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20240908_135426_Joy")

device = "cuda"

messages = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
    {"role": "user", "content": "東京工業大学の主なキャンパスについて教えてください"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])