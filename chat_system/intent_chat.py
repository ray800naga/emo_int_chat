import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftConfig, PeftModel
import numpy as np

device = "cuda"

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

emotion_list = [
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
]

# bf16
# intent 1turn
# adapter_path_list = [
# 	"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240908_125838_acknowledging",
# 	"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240909_102133_agreeing",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240910_025103_consoling",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240910_230527_encouraging",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240911_144251_questioning",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240921_200113_suggesting",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240912_190737_sympathizing",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240914_034959_wishing",
# ]

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

# # emotion 3turn low_lr
# adapter_path_list = [
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20240926_013812_Joy",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20240927_124104_Sadness",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20240929_105440_Anticipation",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20241001_032448_Surprise",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20241002_161255_Anger",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20241004_103056_Fear",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20241006_032625_Disgust",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/emotion_lora_tuning/tuned_model/emotion_lora_Swallow-7b-instruct-v0.1_20241007_120651_Trust"
# ]

# 4bit
# adapter_path_list = [
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/4bit/intent_lora_Swallow-7b-instruct-v0.1_20240910_023135_agreeing_4bit",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/4bit/intent_lora_Swallow-7b-instruct-v0.1_20240911_073941_consoling_4bit",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/4bit/intent_lora_Swallow-7b-instruct-v0.1_20240912_191451_encouraging_4bit",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/4bit/intent_lora_Swallow-7b-instruct-v0.1_20240913_211020_questioning_4bit",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/4bit/intent_lora_Swallow-7b-instruct-v0.1_20240915_062056_suggesting_4bit",
#     "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/4bit/intent_lora_Swallow-7b-instruct-v0.1_20240918_054129_wishing_4bit"
# ]

import pickle
from tqdm import tqdm

with open(
    # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue.pkl",
    # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_1turn.pkl",
    "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_3turn.pkl",
    "rb",
) as f:
    conversation_list = pickle.load(f)
conversation_prompt_list = []
for conversation in tqdm(conversation_list):
    conversation.insert(
        0,
        {
            "role": "system",
            "content": "ユーザの発話に対して共感し、寄り添うような返答を日本語でしてください。その際、一言から二言程度で短く端的に答えてください。",
        },
    )
    conversation_prompt_list.append(conversation)

# base_modelの出力
# base_model = AutoModelForCausalLM.from_pretrained("/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1", device_map="auto").eval()
# tokenizer = AutoTokenizer.from_pretrained("/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1")
# print("base_model")
# for messages in conversation_prompt_list[:50]:
#     encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

#     model_inputs = encodeds.to(device)
#     # model.to(device)

#     generated_ids = base_model.generate(model_inputs, max_new_tokens=512, do_sample=True)
#     decoded = tokenizer.batch_decode(generated_ids)
#     print(decoded[0])

# del base_model


# # adapterをつけた時の出力
# for adapter_path in adapter_path_list:
#     base_model = AutoModelForCausalLM.from_pretrained("/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1", device_map="auto").eval()
#     tokenizer = AutoTokenizer.from_pretrained("/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1")
#     model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
#     model.merge_adapter()

#     print(adapter_path.split('_')[-1])

#     for messages in conversation_prompt_list[:5]:
#         encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

#         model_inputs = encodeds.to(device)
#         # model.to(device)

#         generated_ids = model.generate(model_inputs, max_new_tokens=128, do_sample=True)
#         decoded = tokenizer.batch_decode(generated_ids)
#         print(decoded[0])
#     del model
#     del base_model


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


# adapter selectionの機構を導入
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

# no weight
# intent_predict_model = AutoModelForSequenceClassification.from_pretrained("/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20240803_040940_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-9022", num_labels=8)
# predict_model_tokenizer = AutoTokenizer.from_pretrained("/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20240803_040940_bert-base-japanese-v3_reduce_lr_on_plateau")

# # weight
intent_predict_model = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-67665",
    device_map="cuda",
)
predict_model_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau"
)

output_intent_counter = {}
for intent in intent_list:
    output_intent_counter[intent] = []

for conversation in conversation_list:
    user_input = conversation[-1]["content"]
    tokens = predict_model_tokenizer(user_input, truncation=True, return_tensors="pt")
    tokens = {
        key: value.to(intent_predict_model.device) for key, value in tokens.items()
    }
    preds = intent_predict_model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    out_dict = {n: p for n, p in zip(intent_list, prob)}
    output_intent_counter[max(out_dict, key=out_dict.get)].append(
        (user_input, out_dict)
    )
    # sort
    out_dict_sorted = sorted(out_dict.items(), reverse=True, key=lambda x: x[1])
    # select adapter
    threshold_of_adapter_selection = 0.8  # threshold of adapter selection
    selected_adapter_list = []
    prob_sum = 0
    last_prob_score = 0
    for adapter_score in out_dict_sorted:
        selected_adapter_list.append(adapter_score[0])
        prob_sum += adapter_score[1]
        last_prob_score = adapter_score[1]
        if prob_sum > threshold_of_adapter_selection:
            break
    weights = []

    for intent in intent_list:
        if intent in selected_adapter_list:
            weights.append(out_dict[intent])
        else:
            weights.append(0.0)

    # divide by min_prob
    min_value = last_prob_score
    weights = [x / min_value for x in weights]

    adapter_name = "merge"
    density = 0.2
    model.add_weighted_adapter(
        adapters, weights, adapter_name, combination_type="ties", density=density
    )

    model.set_adapter("merge")

    # input conversation to LLM
    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)
    print("#######")
    print(selected_adapter_list)
    print(weights)
    print(tokenizer.decode(outputs[0]))
    input("press enter to continue...")


# for intent in intent_list:
#     print(f"{intent}: {len(output_intent_counter[intent])}")
#     print(output_intent_counter[intent][:5])


# print(output_intent_counter)
