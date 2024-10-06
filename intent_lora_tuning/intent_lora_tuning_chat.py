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

# random seed の設定
seed = 0

torch.manual_seed(seed)

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


def build_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # pickleの読み込み
    with open(
        # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue.pkl",
        # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_1turn.pkl",
        "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_3turn.pkl",
        "rb",
    ) as f:
        conversation_list = pickle.load(f)
    dic = {"input_ids": [], "query": []}
    for conversation in tqdm(conversation_list):
        conversation.insert(0, {"role": "system", "content": "ユーザの発話に対して共感し、寄り添うような返答を日本語でしてください。その際、一言から二言程度で短く端的に答えてください。"})
        encoded = tokenizer.apply_chat_template(conversation)
        decoded = tokenizer.decode(encoded)
        dic["input_ids"].append(encoded)
        dic["query"].append(decoded)
    df = pd.DataFrame(dic)
    ds = Dataset.from_pandas(df)
    ds.set_format(type="torch")
    # with open(
    #     "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_Dataset.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(ds, f)
    return ds


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
        wandb.init(
            project=f"intent_lora_tuning",
            name=f"intent_lora_{config.model_name.split('/')[-1]}_{current_time}_{intent}_3turn_low_lr",
        )

        dataset = build_dataset(config)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )

        # Load LLM models
        lora_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            # config.model_name, device_map="auto", peft_config=peft_config, load_in_4bit=True
            config.model_name, device_map="auto", peft_config=peft_config
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        tokenizer.pad_token = tokenizer.eos_token

        # initialize PPOTrainer (set ref_model as None)
        ppo_trainer = PPOTrainer(
            config, lora_model, None, tokenizer, dataset=dataset, data_collator=collator
        )

        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"

        reward_model_path = "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/20240729_022849_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-13674"

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

        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]

            #### Get response from llm
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
            

            #### Compute sentiment score
            texts = batch["response"]
            pipe_outputs = emotion_pipe(texts, **sent_kwargs)
            rewards = []
            for output in pipe_outputs:
                for label_score in output:
                    if label_score["label"] == intent:
                        rewards.append(torch.tensor(label_score["score"]))
                        break
            

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        # #### get a batch from the dataset
        # bs = 32
        # game_data = dict()
        # dataset.set_format("pandas")
        # df_batch = dataset[:].sample(bs)
        # game_data["query"] = df_batch["query"].tolist()
        # query_tensors = df_batch["input_ids"].tolist()

        # response_tensors_ref, response_tensors = [], []

        # #### get response from gpt2 and gpt2_ref
        # for i in range(bs):
        # 	# gen_len = output_length_sampler()
        # 	gen_len = 20
        # 	output = ref_model.generate(
        # 		torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
        # 	).squeeze()[-gen_len:]
        # 	response_tensors_ref.append(output)
        # 	output = lora_model.generate(
        # 		torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
        # 	).squeeze()[-gen_len:]
        # 	response_tensors.append(output)

        # #### decode responses
        # game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
        # game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

        # #### sentiment analysis of query/response pairs before/after
        # texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
        # game_data["rewards (before)"] = [output[intent_id]["score"] for output in emotion_pipe(texts, **sent_kwargs)]

        # texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
        # game_data["rewards (after)"] = [output[intent_id]["score"] for output in emotion_pipe(texts, **sent_kwargs)]

        # # store results in a dataframe
        # df_results = pd.DataFrame(game_data)

        # print("mean:")
        # print(df_results[["rewards (before)", "rewards (after)"]].mean())
        # print()
        # print("median:")
        # print(df_results[["rewards (before)", "rewards (after)"]].median())

        save_dir = f"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_{config.model_name.split('/')[-1]}_{current_time}_{intent}"
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
        # "acknowledging",
        # "agreeing",
        # "consoling",
        # "encouraging",
        # "questioning",
        # "suggesting",
        "sympathizing",
        # "wishing",
    ]
    for intent in intent_list[:]:
        main(intent)
    send_slack_message("All training completed succesfully. (intent lora)")
