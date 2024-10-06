import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, load_metric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
import json
import os
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import wandb

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from scheduler import my_get_scheduler

# モデル指定
# model_name = 'bert-large-japanese-v2'
model_name = 'bert-base-japanese-v3'

# Slack通知の設定
with open("/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/slack_API.txt", 'r') as f:
    slack_token = f.read().strip()

client = WebClient(token=slack_token)

# 自分のユーザーIDを取得
with open("/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/slack_user_id.txt") as f:
    slack_user = f.read().strip()

def send_slack_message(message):
    try:
        res_conv_open = client.conversations_open(users=slack_user)
        dm_id = res_conv_open['channel']['id']
        response = client.chat_postMessage(
            channel=dm_id,
            text=message
        )
        assert response["ok"]
    except SlackApiError as e:
        print(f"Slack API Error: {e.response['error']}")

def main(scheduler_name):
    try:
        # 現在の日時を取得してフォーマット
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # empathetic intent with next labelの読み込み
        df = pd.read_excel('/workspace/Emotion_Intent_Chat/empathetic_dialogues/empathetic_intent_jp_next_label.xlsx', sheet_name=None)

        # 8つの発話意図のリスト
        intent_names = ["questioning", "acknowledging", "consoling", "agreeing", "encouraging", "sympathizing", "suggesting", "wishing"]

        df_all = pd.DataFrame()
        for sheet in df.keys():
            df_all = pd.concat([df_all, df[sheet]])

        # Labelがintentで、Typeがutteranceの者だけに絞り込む
        df_all = df_all[df_all["Next_Label"].isin(intent_names)]
        df_all = df_all[df_all["Type"] == "utterance"]

        df_all = df_all.loc[:, ["テキスト", "Next_Label"]]
        df_all = df_all.rename(columns={"テキスト": "text", "Next_Label": "label_str"})

        # # for debug
        # df_all = df_all[:100]


        # ラベルエンコーディング
        le = LabelEncoder()
        df_all["label"] = le.fit_transform(df_all["label_str"])
        print(le.classes_)  # ここでラベルの対応関係が表示される

        df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=0)

        checkpoint = f'/workspace/Emotion_Intent_Chat/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def tokenize_function(batch):
            tokenized_batch = tokenizer(batch['text'], truncation=True, padding="max_length")
            return tokenized_batch

        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)


        num_labels = 8
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

        # 評価指標を定義
        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = metric.compute(predictions=predictions, references=labels)
            wandb.log({"accuracy": accuracy['accuracy']})
            return accuracy

        # 訓練時の設定
        output_dir = f'tuned_model/{current_time}_{model_name}_{scheduler_name}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            num_train_epochs=30,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=5e-6,  # 初期の学習率を指定
            warmup_ratio=0.05,
            report_to="wandb"  # Weight & Biasesへログを送信するように設定
        )

        # W&Bの初期化
        # wandb.init(project=f"intent_reward_model_{model_name}", config=training_args, name=f"intent_{current_time}_{model_name}")

        # for debug
        wandb.init(project=f"next_intent_prediction_model_{model_name}", config=training_args, name=f"intent_prediction_{current_time}_{model_name}_{scheduler_name}")

        # オプティマイザの設定
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

        # スケジューラの設定
        num_training_steps = (len(train_tokenized_dataset) // (training_args.per_device_train_batch_size * torch.cuda.device_count())) * training_args.num_train_epochs
        lr_scheduler = my_get_scheduler(
            scheduler_name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps
        )

        # Trainerの設定
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=test_tokenized_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler)  # オプティマイザとスケジューラをTrainerに渡す
        )

        # 学習実行
        trainer.train()

        # モデルとトークナイザーの保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # ラベル情報をlabel_id.jsonに保存
        label_mapping = {int(i): label for i, label in enumerate(le.classes_)}
        label_path = os.path.join(output_dir, "label_id.json")
        with open(label_path, "w") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=4)

        print("Saved label mapping to label_id.json:")
        print(label_mapping)
        
        wandb.finish()
        send_slack_message(f"Training completed successfully. \r{output_dir} \rintent_prediction_{current_time}_{model_name}_{scheduler_name}")
    except Exception as e:
        send_slack_message(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    scheduler_name_list = ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup', 'reduce_lr_on_plateau']
    for scheduler_name in scheduler_name_list[-1:]:
        print(f"Start training with {scheduler_name} scheduler")
        main(scheduler_name)
        print(f"Finish training with {scheduler_name} scheduler")
    send_slack_message("All training completed successfully.")
