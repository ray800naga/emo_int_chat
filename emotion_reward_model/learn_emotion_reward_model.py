import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, get_scheduler, EarlyStoppingCallback
from datasets import Dataset, load_metric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import wandb
from scipy.spatial.distance import cosine

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from scheduler import my_get_scheduler

# モデル指定
model_name = 'bert-base-japanese-v3'
# model_name = 'bert-large-japanese-v2'

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

        df_wrime = pd.read_table("/workspace/Emotion_Intent_Chat/wrime/wrime-ver1.tsv")

        # 8感情のリスト
        emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
        emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
        num_labels = len(emotion_names)

        df_wrime['writer_emotion_intensities'] = df_wrime.apply(lambda x: [x['Writer_' + name] for name in emotion_names], axis=1)

        df_wrime_target = df_wrime[df_wrime['writer_emotion_intensities'].apply(lambda x: x != [0,0,0,0,0,0,0,0])]

        # for debug
        print(f"df_wrime: {len(df_wrime)}, df_wrime_target: {len(df_wrime_target)}")

        # train / testに分割
        df_groups = df_wrime_target.groupby('Train/Dev/Test')
        df_train = df_groups.get_group('train')
        df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])

        # for debug
        # df_train = df_train[:100]
        # df_test = df_test[:20]

        checkpoint = f'/workspace/Emotion_Intent_Chat/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # token化、writer_emotion_intensitiesをlabelsとして扱う
        def tokenize_function(batch):
            tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding="max_length")
            tokenized_batch['labels'] = np.array(batch['writer_emotion_intensities']).astype(float)
            return tokenized_batch

        # transformers用のデータセット形式に変換
        target_columns = ['Sentence', 'writer_emotion_intensities']
        train_dataset = Dataset.from_pandas(df_train[target_columns])
        test_dataset = Dataset.from_pandas(df_test[target_columns])

        # 前処理の適用
        train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, problem_type="regression")

        # コサイン類似度の計算関数を定義
        def cosine_similarity(a, b):
            return 1 - cosine(a, b)

        # 評価指標を定義
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # sigmoid = torch.nn.Sigmoid()
            # probs = sigmoid(torch.tensor(logits))
            probs = torch.tensor(logits)
            probs = probs.detach().numpy()

            cosine_similarities = [cosine_similarity(pred, true) for pred, true in zip(probs, labels)]
            avg_cosine_similarity = np.mean(cosine_similarities)
            wandb.log({"cosine_similarity": avg_cosine_similarity})
            return {"cosine_similarity": avg_cosine_similarity}

        # 訓練時の設定
        output_dir = f'tuned_model/{current_time}_{model_name}_{scheduler_name}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            num_train_epochs=15,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=5e-6,  # 初期の学習率を指定
            warmup_ratio=0.05,
            report_to="wandb"  # Weight & Biasesへログを送信するように設定
        )

        # W&Bの初期化
        wandb.init(project=f"cos_sim_fixed_writer_intensity_emotion_reward_model_{model_name}", config=training_args, name=f"{current_time}_{model_name}_{scheduler_name}")

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
        label_mapping = {int(i): label for i, label in enumerate(emotion_names)}
        label_path = os.path.join(output_dir, "label_id.json")
        with open(label_path, "w") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=4)

        print("Saved label mapping to label_id.json:")
        print(label_mapping)

        wandb.finish()
        send_slack_message(f"Training completed successfully. \r{output_dir}")
    except Exception as e:
        send_slack_message(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    scheduler_name_list = ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup', 'reduce_lr_on_plateau']
    for scheduler_name in scheduler_name_list:
        print(f"Start training with {scheduler_name} scheduler")
        main(scheduler_name)
        print(f"Finish training with {scheduler_name} scheduler")
    send_slack_message("All training completed successfully.")
