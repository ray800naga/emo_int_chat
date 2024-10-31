import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import xlora
from tqdm import tqdm
import pickle
import pandas as pd
from datasets import Dataset
from peft import PeftModel
import wandb
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime
import os

# Slack通知の設定
with open(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/slack_API.txt", "r"
) as f:
    slack_token = f.read().strip()

client = WebClient(token=slack_token)

# 自分のユーザーIDを取得
with open(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/slack_user_id.txt"
) as f: 
    slack_user = f.read().strip()
    
def send_slack_message(message):
    try:
        res_conv_open = client.conversations_open(users=slack_user)
        dm_id = res_conv_open["channel"]["id"]
        response = client.chat_postMessage(channel=dm_id, text=message)
        assert response["ok"]
    except SlackApiError as e:
        print(f"Slack API Error: {e.response['error']}")

intent_list = [
    "acknowleding",
    "agreeing",
    "consoling",
    "encouraging",
    "questioning",
    "suggesting",
    "sympathizing",
    "wishing",
]

# intent 3turn low_lr
adapter_path_list = [
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240925_064514_acknowledging",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240926_093351_agreeing",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240927_193209_consoling",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240929_014449_encouraging",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240930_035253_questioning",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20241001_020956_suggesting",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20241005_084716_sympathizing",
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20241003_163251_wishing",
]

# コサイン類似度損失
def cosine_similarity_loss(vec1, vec2):
    cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
    loss = 1 - cos_sim
    return loss.mean()

# Intentベクトルを取得
def get_intent_vector(text, model, tokenizer):
    tokens = tokenizer(text, truncation=True, return_tensors="pt").to(device)
    outputs = model(**tokens)
    return F.softmax(outputs.logits, dim=-1)

def build_dataset(llm_tokenizer):
    with open(
        # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue.pkl",
        # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_1turn.pkl",
        "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_3turn.pkl",
        "rb",
    ) as f:
        conversation_list = pickle.load(f)
        
    # # for debug
    # conversation_list = conversation_list[0:80]
    
    dic = {"input_ids": [], "query": [], "last_user_input": []}
    for conversation in tqdm(conversation_list):
        conversation.insert(0, {"role": "system", "content": "ユーザの発話に対して共感し、寄り添うような返答を日本語でしてください。その際、一言から二言程度で短く端的に答えてください。"})
        encoded = llm_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
        decoded = llm_tokenizer.decode(encoded)
        dic["input_ids"].append(encoded)
        dic["query"].append(decoded)
        dic["last_user_input"].append(conversation[-1]["content"])
    return dic

# 学習ループの定義
def train_model(model, tokenizer, intent_predict_model, intent_analyze_model, dataset, optimizer, scaling_weights_model, epochs=3, batch_size=8):
    try:
        # 学習ループ
        for epoch in range(epochs):
            epoch_loss = 0
            dataset_num = len(dataset["input_ids"])
            for batch_idx, batch_head_index in enumerate(tqdm(range(0, dataset_num, batch_size), desc=f"Epoch {epoch + 1}/{epochs}")):
                last_user_input_list = dataset["last_user_input"][batch_head_index:batch_head_index+batch_size]
                input_ids_list = dataset["input_ids"][batch_head_index:batch_head_index+batch_size]
                query_list = dataset["query"][batch_head_index:batch_head_index+batch_size]
                # Next Intent Predictモデルを使ってIntentベクトルを取得
                batch_intent_vectors = []
                for user_input in last_user_input_list:
                    intent_vector = get_intent_vector(user_input, intent_predict_model, next_intent_predict_tokenizer)
                    batch_intent_vectors.append(intent_vector)
                batch_intent_vectors = torch.stack(batch_intent_vectors).to(device)  # スタックして8次元ベクトルに

                # ScalingWeightsNNを使ってscaling_weightsを計算
                scaling_weights = []
                for user_input in last_user_input_list:
                    tokens = next_intent_predict_tokenizer(user_input, return_tensors="pt").to(device)
                    outputs = scaling_weights_model(**tokens).logits
                    scaling_weights.append(F.softmax(outputs, dim=-1))
                scaling_weights = torch.stack(scaling_weights).squeeze(1).to(device)  # スケーリングウェイトをスタックしてバッチ化

                # LLMによるテキスト生成（scaling_weightsを使用してアダプタを適用）
                generated_texts = []
                query_tensors = [torch.tensor(q).to(device) for q in input_ids_list]  # 各要素をテンソルに変換
                for i, query_tensor in enumerate(query_tensors):
                    model.add_weighted_adapter(adapters=adapters, weights=scaling_weights[i], combination_type="ties", adapter_name="weighted", density=0.2)
                    model.set_adapter("weighted")
                    generated_outputs = model.generate(input_ids=query_tensor.unsqueeze(0), max_new_tokens=256, do_sample=True)
                    generated_text = tokenizer.decode(generated_outputs[0])
                    generated_texts.append(generated_text[len(query_list[i]):])

                # 生成されたテキストに対してIntentを予測
                generated_intent_vectors = []
                for generated_text in generated_texts:
                    generated_intent_vector = get_intent_vector(generated_text, intent_analyze_model, intent_analyze_tokenizer)
                    generated_intent_vectors.append(generated_intent_vector)
                generated_intent_vectors = torch.stack(generated_intent_vectors).to(device)  # Intentの予測結果（8次元）

                # コサイン類似度を使って損失を計算
                loss = cosine_similarity_loss(generated_intent_vectors, batch_intent_vectors)

                # 勾配を初期化し、逆伝播で最適化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ロスを集計
                epoch_loss += loss.item()
                
                # ステップごとに損失をWandBにログ
                wandb.log({"step_loss": loss.item(), "epoch": epoch + 1, "step": batch_idx + 1})

            # エポックごとに損失をWandBにログ
            avg_epoch_loss = epoch_loss / (len(dataset["input_ids"]) / batch_size)
            wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss})
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / (len(dataset['input_ids'])/batch_size)}")
            
            # モデルをエポックごとに保存
            model_save_path = f"/workspace/Emotion_Intent_Chat/emo_int_chat/mixture_lora_weight/tuned_model/lora_mixture_weight_{current_time}/"
            os.makedirs(model_save_path, exist_ok=True)
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': scaling_weights_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f"{model_save_path}lora_mixture_weight_{epoch+1}.pth")
            print(f"Model saved at {model_save_path}lora_mixture_weight_BERT_epoch{epoch+1}.pth")
        wandb.finish()
    except Exception as e:
        wandb.finish()
        send_slack_message(f"Training Failed with error: {str(e)}")
        


# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# Intent予測モデル
next_intent_predict_model = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-67665"
)
next_intent_predict_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau"
)

# Intent分析モデル
intent_analyze_model = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/20240729_022849_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-13674"
)
intent_analyze_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/20240729_022849_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-13674"
)

next_intent_predict_model = next_intent_predict_model.to(device)
intent_analyze_model = intent_analyze_model.to(device)

# LLM model定義
base_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1",
    device_map="auto",
    load_in_8bit=False,
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1"
)

model = PeftModel.from_pretrained(
    base_model, adapter_path_list[0], adapter_name=intent_list[0]
)
for i, adapter in enumerate(adapter_path_list):
    _ = model.load_adapter(adapter, adapter_name=intent_list[i])


adapters = intent_list

# cache利用をFalseに設定
model.config.use_cache = False

# ScalingWeightsBERTの定義
scaling_weights_model = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/Emotion_Intent_Chat/bert-base-japanese-v3",
    num_labels=8
)
scaling_weights_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/bert-base-japanese-v3"
)

scaling_weights_model = scaling_weights_model.to(device)

# オプティマイザの設定
optimizer = torch.optim.Adam(scaling_weights_model.parameters(), lr=1e-3)

# データセットの作成
dataset = build_dataset(tokenizer)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(
    project=f"lora_mixture_weight",
    name=f"lora_mixture_weight_BERT{current_time}",
)

# モデルの学習
train_model(
    model=model,
    tokenizer=tokenizer,
    intent_predict_model=next_intent_predict_model,
    intent_analyze_model=intent_analyze_model,
    dataset=dataset,
    optimizer=optimizer,
    scaling_weights_model=scaling_weights_model,
    epochs=10,  # 学習エポック数
    batch_size=8  # バッチサイズ
)

send_slack_message("learn lora mixture weight: all training completed successfully.")