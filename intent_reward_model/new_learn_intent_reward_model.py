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
from torch.nn import CrossEntropyLoss
from multiprocessing import Pool

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from scheduler import my_get_scheduler

# モデル指定
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
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # empathetic intentの読み込み
        df = pd.read_excel('/workspace/Emotion_Intent_Chat/empathetic_dialogues/empathetic_intent_jp_2nd.xlsx', sheet_name=None)

        intent_names = ["questioning", "acknowledging", "consoling", "agreeing", "encouraging", "sympathizing", "suggesting", "wishing"]

        df_all = pd.DataFrame()
        for sheet in df.keys():
            df_all = pd.concat([df_all, df[sheet]])

        df_all = df_all[df_all["Label"].isin(intent_names)]
        df_all = df_all[df_all["Type"] == "utterance"]

        df_all = df_all.loc[:, ["テキスト", "Label"]]
        df_all = df_all.rename(columns={"テキスト": "text", "Label": "label_str"})

        # ラベルエンコーディング
        le = LabelEncoder()
        df_all["label"] = le.fit_transform(df_all["label_str"])
        print(le.classes_)

        df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=0)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=0)

        checkpoint = f'/workspace/Emotion_Intent_Chat/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def tokenize_function(batch):
            return tokenizer(batch['text'], truncation=True, padding="max_length")

        train_dataset = Dataset.from_pandas(df_train)
        val_dataset = Dataset.from_pandas(df_val)
        test_dataset = Dataset.from_pandas(df_test)

        train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        val_tokenized_dataset = val_dataset.map(tokenize_function, batched=True)
        test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

        num_labels = 8
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

        class_counts = df_train['label'].value_counts().sort_index().values
        class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

        loss_fn = CrossEntropyLoss(weight=class_weights)

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss = loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

        accuracy_metric = load_metric("accuracy")
        f1_metric = load_metric("f1")
        recall_metric = load_metric("recall")
        precision_metric = load_metric("precision")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
            recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')
            precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')
            
            wandb.log({
                "accuracy": accuracy['accuracy'],
                "f1": f1['f1'],
                "recall": recall['recall'],
                "precision": precision['precision']
            })
            
            return {
                "accuracy": accuracy['accuracy'],
                "f1": f1['f1'],
                "recall": recall['recall'],
                "precision": precision['precision']
            }

        output_dir = f'tuned_model/weighted_{current_time}_{model_name}_{scheduler_name}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            num_train_epochs=30,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=5e-6,
            warmup_ratio=0.05,
            report_to="wandb"
        )

        wandb.init(project=f"weighted_intent_reward_model_{model_name}", config=training_args, name=f"intent_{current_time}_{model_name}_{scheduler_name}")

        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

        num_training_steps = (len(train_tokenized_dataset) // (training_args.per_device_train_batch_size * torch.cuda.device_count())) * training_args.num_train_epochs
        lr_scheduler = my_get_scheduler(
            scheduler_name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=val_tokenized_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler)
        )

        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        label_mapping = {int(i): label for i, label in enumerate(le.classes_)}
        label_path = os.path.join(output_dir, "label_id.json")
        with open(label_path, "w") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=4)

        print("Saved label mapping to label_id.json:")
        print(label_mapping)

        test_results = trainer.evaluate(eval_dataset=test_tokenized_dataset)
        print("Test results:", test_results)

        wandb.log({
            "test_accuracy": test_results['eval_accuracy'],
            "test_f1": test_results['eval_f1'],
            "test_recall": test_results['eval_recall'],
            "test_precision": test_results['eval_precision']
        })
        
        wandb.finish()
        send_slack_message(f"Training completed successfully. \r{output_dir}")
    except Exception as e:
        send_slack_message(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    scheduler_name_list = ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup', 'reduce_lr_on_plateau']
    
    with Pool(processes=1) as pool:
        pool.map(main, scheduler_name_list)
    send_slack_message("All training completed successfully.")
