import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda"

messages = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
    {"role": "user", "content": "東京工業大学の主なキャンパスについて教えてください"}
]

# メッセージをエンコードし、テンソル形式で取得
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# 入力をcudaに転送
model_inputs = encodeds['input_ids'].to(device)

# モデルをデバイスに移動
model.to(device)

# 生成
generated_ids = model.generate(model_inputs, max_new_tokens=128, do_sample=True)

# 出力をデコード
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
