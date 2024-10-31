import torch
import xlora
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import pandas as pd
from datasets import Dataset

def cosine_similarity_loss(vec1, vec2):
    cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
    loss = 1 - cos_sim
    return loss.mean()

def get_intent_vector(text, model, tokenizer):
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    outputs = model(**tokens)
    return F.softmax(outputs.logits, dim=-1)

def build_dataset(llm_tokenizer, next_intent_predict_model, next_intent_predict_tokenizer):
    with open(
        # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue.pkl",
        "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_1turn.pkl",
        # "/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_3turn.pkl",
        "rb",
    ) as f:
        conversation_list = pickle.load(f)
    dic = {"input_ids": [], "query": [], "next_intent_predict_vector": []}
    for conversation in tqdm(conversation_list):
        conversation.insert(0, {"role": "system", "content": "ユーザの発話に対して共感し、寄り添うような返答を日本語でしてください。その際、一言から二言程度で短く端的に答えてください。"})
        encoded = llm_tokenizer.apply_chat_template(conversation)
        decoded = llm_tokenizer.decode(encoded)
        dic["input_ids"].append(encoded)
        dic["query"].append(decoded)
        text = conversation[-1]["content"]
        tokens = next_intent_predict_tokenizer(text, truncation=True, return_tensors="pt")
        preds = next_intent_predict_model(**tokens)
        dic["next_intent_predict_vector"].append = F.softmax(preds.logits, dim=-1)
    df = pd.DataFrame(dic)
    ds = Dataset.from_pandas(df)
    ds.set_format(type="torch")
    return ds
    
        

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1",
    device_map="auto",
    load_in_8bit=False,
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/Swallow-7b-instruct-v0.1"
)

next_intent_predict_model = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-67665",
    device_map="auto"
)
next_intent_predict_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/next_intent_predict_model/tuned_model/20241006_161421_bert-base-japanese-v3_reduce_lr_on_plateau"
)

intent_analyze_model = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/20240729_022849_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-13674",
    device_map="auto"
)
intent_analyze_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/20240729_022849_bert-base-japanese-v3_reduce_lr_on_plateau/checkpoint-13674"
)

xlora_config = xlora.xLoRAConfig(
    hidden_size=model.config.hidden_size,
    base_model_id="mistralai/Mistral-7B-Instruct-v0.1",
    xlora_depth=8,
    device=torch.device("auto"),
    adapters={
        "acknowleding": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240925_064514_acknowledging",
        "agreeing": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240926_093351_agreeing",
        "consoling": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240927_193209_consoling",
        "encouraging": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240929_014449_encouraging",
        "questioning": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20240930_035253_questioning",
        "suggesting": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20241001_020956_suggesting",
        "sympathizing": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20241005_084716_sympathizing",
        "wishing": "/workspace/Emotion_Intent_Chat/emo_int_chat/intent_lora_tuning/tuned_model/intent_lora_Swallow-7b-instruct-v0.1_20241003_163251_wishing"
    },
)

model_created = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora_config,
    verbose=True,
)

# define scaling weight
scaling_weights = torch.nn.Parameter(torch.ones(len(xlora_config.adapters), requires_grad=True, device="auto"))

# optimizer configuration
optimizer = torch.optim.Adam([scaling_weights], lr=1e-3)

dataset = build_dataset(tokenizer, next_intent_predict_model, next_intent_predict_tokenizer)



