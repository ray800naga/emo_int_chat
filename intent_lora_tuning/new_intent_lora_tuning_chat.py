import torch
from tqdm import tqdm
import pandas as pd
import pickle

tqdm.pandas()
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import os
import wandb
from datetime import datetime
from peft import LoraConfig, TaskType
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import json
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# random seed の設定
seed = 0
torch.manual_seed(seed)

# Slack通知の設定
with open(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/slack_API.txt", "r"
) as f:
    slack_token = f.read().strip()
client = WebClient(token=slack_token)
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


def build_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    with open(
        "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_3turn.pkl",
        "rb",
    ) as f:
        conversation_list = pickle.load(f)
    dic = {"input_ids": [], "query": [], "last_user_input": [], "conversation": []}
    for conversation in tqdm(conversation_list):
        conversation.insert(
            0,
            {
                "role": "system",
                "content": "ユーザの発話に対して共感し、寄り添うような返答を日本語でしてください。その際、一言から二言程度で短く端的に答えてください。",
            },
        )
        encoded = tokenizer.apply_chat_template(conversation)
        decoded = tokenizer.decode(encoded)
        dic["input_ids"].append(encoded)
        dic["query"].append(decoded)
        dic["last_user_input"].append(conversation[-1]["content"])
        dic["conversation"].append(conversation)
    df = pd.DataFrame(dic)

    # Split the dataset
    train_df, val_test_df = train_test_split(df, test_size=0.05, random_state=seed)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=seed)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")
    test_ds.set_format(type="torch")

    with open("3turns_test.pkl", "wb") as f:
        pickle.dump(test_ds, f)

    return train_ds, val_ds, test_ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(intent):
    try:
        config = PPOConfig(
            model_name="/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1",
            learning_rate=1.41e-6,
            log_with="wandb",
            batch_size=8,
            mini_batch_size=2,
            gradient_accumulation_steps=4,
        )

        sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # wandb.init(
        #     project=f"intent_lora_tuning_new_reward_model",
        #     name=f"weighted_intent_lora_{config.model_name.split('/')[-1]}_{current_time}_{intent}_3turn_low_lr",
        # )

        train_ds, val_ds, test_ds = build_dataset(config)
        exit()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )

        lora_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name, device_map="auto", peft_config=peft_config
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        ppo_trainer = PPOTrainer(
            config,
            lora_model,
            None,
            tokenizer,
            dataset=train_ds,
            data_collator=collator,
        )

        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"

        reward_model_path = "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/weighted_20241104_110732_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-68370"
        emotion_pipe = pipeline(
            "text-classification", model=reward_model_path, device=device
        )
        reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        with open(
            os.path.join(reward_model_path, "label_id.json"),
            mode="rt",
            encoding="utf-8",
        ) as f:
            intent_dict = json.load(f)
        for k, v in intent_dict.items():
            if v == intent:
                intent_id = int(k)
                break

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 128,
        }

        for batch_num, batch in tqdm(
            enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)
        ):
            query_tensors = batch["input_ids"]

            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                query_len = len(query.squeeze())
                response_tensors.append(response.squeeze()[query_len:])
            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]
            reward_tokenized_response = [
                reward_model_tokenizer.encode(r) for r in batch["response"]
            ]
            for i, tokens in enumerate(reward_tokenized_response):
                if len(tokens) > 512:
                    batch["response"][i] = reward_model_tokenizer.decode(tokens[:500])

            texts = batch["response"]
            pipe_outputs = emotion_pipe(texts, **sent_kwargs)
            rewards = []
            for output in pipe_outputs:
                for label_score in output:
                    if label_score["label"] == intent:
                        rewards.append(torch.tensor(label_score["score"]))
                        break

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            # Validation step every 125 batches
            if batch_num % 125 == 0 and batch_num != 0:
                val_rewards = []
                for val_data in tqdm(val_ds):
                    val_query = val_data["input_ids"]
                    val_response = ppo_trainer.generate(val_query, **generation_kwargs)
                    val_query_len = len(val_query.squeeze())
                    val_response_tensor = val_response.squeeze()[val_query_len:]
                    val_response_text = tokenizer.decode(val_response_tensor)
                    val_tokenized_response = reward_model_tokenizer.encode(
                        val_response_text
                    )
                    if len(val_tokenized_response) > 512:
                        val_response_text = reward_model_tokenizer.decode(
                            val_tokenized_response[:500]
                        )
                    val_pipe_output = emotion_pipe(val_response_text, **sent_kwargs)
                    for label_score in val_pipe_output:
                        if label_score["label"] == intent:
                            val_rewards.append(torch.tensor(label_score["score"]))
                            break
                wandb.log(
                    {"validation_reward_mean": torch.mean(torch.stack(val_rewards))}
                )

        # Test evaluation
        test_rewards = []
        for test_data in tqdm(test_ds):
            test_query = test_data["input_ids"]
            test_response = ppo_trainer.generate(test_query, **generation_kwargs)
            test_query_len = len(test_query.squeeze())
            test_response_tensor = test_response.squeeze()[test_query_len:]
            test_response_text = tokenizer.decode(test_response_tensor)
            test_tokenized_response = reward_model_tokenizer.encode(test_response_text)
            if len(test_tokenized_response) > 512:
                test_response_text = reward_model_tokenizer.decode(
                    test_tokenized_response[:500]
                )
            test_pipe_output = emotion_pipe(test_response_text, **sent_kwargs)
            for label_score in test_pipe_output:
                if label_score["label"] == intent:
                    test_rewards.append(torch.tensor(label_score["score"]))
                    break
        wandb.log({"test_reward_mean": torch.mean(torch.stack(test_rewards))})

        save_dir = f"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/weighted_intent_lora_{config.model_name.split('/')[-1]}_{current_time}_{intent}"
        os.makedirs(save_dir, exist_ok=True)

        lora_model.save_pretrained(save_dir, push_to_hub=False)
        tokenizer.save_pretrained(save_dir, push_to_hub=False)

        wandb.finish()
        send_slack_message(f"Training completed succesfully. \r{save_dir}")

        del lora_model
        del ppo_trainer
        torch.cuda.empty_cache()
    except Exception as e:
        wandb.finish()
        send_slack_message(f"Training failed with error: {str(e)}")
        del lora_model
        del ppo_trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    intent_list = [
        "acknowledging",
        "agreeing",
        "consoling",
        "encouraging",
        "questioning",
        "suggesting",
        "sympathizing",
        "wishing",
    ]
    # for intent in intent_list:
    #     main(intent)

    with Pool(processes=1) as pool:
        pool.map(main, intent_list)
    send_slack_message("All training completed succesfully. (intent lora)")
