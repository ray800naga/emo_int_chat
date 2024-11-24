import pickle
import torch
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	AutoModelForSequenceClassification,
)
import pandas as pd
from tqdm import tqdm
import numpy as np

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

def np_softmax(x):
	f_x = np.exp(x) / np.sum(np.exp(x))
	return f_x

with open("/workspace/Emotion_Intent_Chat/all_turns_test.pkl", 'rb') as f:
	test_ds = pickle.load(f)
	
# # weight
intent_predict_model = AutoModelForSequenceClassification.from_pretrained(
	"/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-67665",
	device_map="cuda",
)
predict_model_tokenizer = AutoTokenizer.from_pretrained(
	"/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau"
)

df_proposed = pd.DataFrame(columns=["messages", "human_response", "mixed_selected_adapter", "single_selected_adapter"])

for conversation in tqdm(test_ds):
	last_user_input = conversation["last_user_input"]
	tokens = predict_model_tokenizer(last_user_input, truncation=True, return_tensors="pt")
	tokens = {
		key: value.to(intent_predict_model.device) for key, value in tokens.items()
	}
	preds = intent_predict_model(**tokens)
	prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
	out_dict = {n: p for n, p in zip(intent_list, prob)}
	# sort
	out_dict_sorted = sorted(out_dict.items(), reverse=True, key=lambda x: x[1])
	# select adapter
	threshold_of_adapter_selection = 0.7  # threshold of adapter selection
	selected_adapter_list = []
	prob_sum = 0
	last_prob_score = 0
	for adapter_score in out_dict_sorted:
		selected_adapter_list.append(adapter_score[0])
		prob_sum += adapter_score[1]
		last_prob_score = adapter_score[1]
		if prob_sum > threshold_of_adapter_selection:
			break

	new_row = pd.DataFrame({"messages": [conversation["conversation"]], "human_response": [conversation["human_response"]], "mixed_selected_adapter": [selected_adapter_list], "single_selected_adapter": [selected_adapter_list[0]]})
	df_proposed = pd.concat([df_proposed, new_row], ignore_index=True)
 
df_proposed.to_excel("final_df_human_output_selected_adapter.xlsx", index=False)