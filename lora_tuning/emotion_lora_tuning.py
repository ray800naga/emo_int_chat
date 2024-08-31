
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

import os

import wandb
from datetime import datetime

from peft import LoraConfig, TaskType

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

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

def build_dataset(config, dataset_name="shunk031/wrime", ver="ver1", input_min_text_length=0, input_max_text_length=8):
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	# tokenizer.pad_token = tokenizer.pad_token
	tokenizer.pad_token = tokenizer.eos_token

	ds = load_dataset(dataset_name, ver, split="train")
	ds = ds.remove_columns(["user_id", "datetime", "writer", "reader1", "reader2", "reader3", "avg_readers"])

	def tokenize(sample):
		stc_length = len(tokenizer.encode(sample["sentence"]))
		if stc_length < input_max_text_length:
			input_size = stc_length
		else :
			input_size = input_max_text_length
		sample["input_ids"] = tokenizer.encode(sample["sentence"])[: input_size]
		sample["query"] = tokenizer.decode(sample["input_ids"])
		return sample

	ds = ds.map(tokenize, batched=False)
	ds.set_format(type="torch")
	return ds

def collator(data):
	return dict((key, [d[key] for d in data]) for key in data[0])

def main(emotion):
	try:
		config = PPOConfig(
			model_name="/workspace/Emotion_Intent_Chat/calm2-7b",
			learning_rate=1.41e-5,
			log_with="wandb",
		)

		sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

		current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
		wandb.init(project=f"test_emotion_lora_tuning", name=f"emotion_lora_{config.model_name.split('/')[-1]}_{current_time}_{emotion}")

		dataset = build_dataset(config)


		peft_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
		)


		# Load LLM models
		lora_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, device_map="auto", peft_config=peft_config)


		tokenizer = AutoTokenizer.from_pretrained(config.model_name)

		tokenizer.pad_token = tokenizer.pad_token


		# initialize PPOTrainer (set ref_model as None)
		ppo_trainer = PPOTrainer(config, lora_model, None, tokenizer, dataset=dataset, data_collator=collator)


		device = ppo_trainer.accelerator.device
		if ppo_trainer.accelerator.num_processes == 1:
			device = 0 if torch.cuda.is_available() else "cpu"



		emotion_pipe = pipeline("text-classification", model="/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_reward_model/tuned_model/20240817_204322_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-4886", device=device)


		emotion_dict = {
			"Joy": 0,
			"Sadness": 1,
			"Anticipation": 2,
			"Surprise": 3,
			"Anger": 4,
			"Fear": 5,
			"Disgust": 6,
			"Trust": 7
		}

		emotion_id = emotion_dict[emotion]


		generation_kwargs = {
			"min_length": -1,
			"top_k": 0.0,
			"top_p": 1.0,
			"do_sample": True,
			"pad_token_id": tokenizer.eos_token_id,
			"max_new_tokens": 32
		}


		for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
			query_tensors = batch["input_ids"]

			#### Get response from calm
			response_tensors = []
			for query in query_tensors:
				response = ppo_trainer.generate(query, **generation_kwargs)
				gen_len = len(response.squeeze())
				response_tensors.append(response.squeeze()[-gen_len:])
			batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

			#### Compute sentiment score
			texts = [q + r for q, r in zip(batch["query"], batch["response"])]
			pipe_outputs = emotion_pipe(texts, **sent_kwargs)
			rewards = [torch.tensor(output[emotion_id]["score"]) for output in pipe_outputs]

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
		# game_data["rewards (before)"] = [output[emotion_id]["score"] for output in emotion_pipe(texts, **sent_kwargs)]

		# texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
		# game_data["rewards (after)"] = [output[emotion_id]["score"] for output in emotion_pipe(texts, **sent_kwargs)]

		# # store results in a dataframe
		# df_results = pd.DataFrame(game_data)


		# print("mean:")
		# print(df_results[["rewards (before)", "rewards (after)"]].mean())
		# print()
		# print("median:")
		# print(df_results[["rewards (before)", "rewards (after)"]].median())



		save_dir = f"/workspace/Emotion_Intent_Chat/emo_int_chat/lora_tuning/tuned_model/emotion_lora_{config.model_name.split('/')[-1]}_{current_time}_{emotion}"
		os.makedirs(save_dir, exist_ok=True)

		lora_model.save_pretrained(save_dir, push_to_hub=False)
		tokenizer.save_pretrained(save_dir, push_to_hub=False)

		wandb.finish()
		send_slack_message(f"Training completed succesfully. \r{save_dir}")

		del(lora_model)
		torch.cuda.empty_cache()
	except Exception as e:
		send_slack_message(f"Training failed with error: {str(e)}")
		wandb.finish()


if __name__ == "__main__":
	emotion_list = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
	for emotion in emotion_list[:]:
		main(emotion)
	send_slack_message("All training completed succesfully. (emotion lora)")
