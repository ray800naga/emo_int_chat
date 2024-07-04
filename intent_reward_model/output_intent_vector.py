import json
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator

model_name = "20240627_141942_bert-base-japanese-v3"
checkpoint = "checkpoint-22785"

# labelデータの取得
label_id_file_path = f"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/{model_name}/label_id.json"
with open(label_id_file_path, 'r', encoding='utf-8') as f:
    label_data = json.load(f)
label_names = []
for i in range(8):
    label_names.append(label_data[str(i)])
print(label_names)

# 日本語出力のための設定
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # フォントパスを指定
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

sns.set_theme()

tokenizer = AutoTokenizer.from_pretrained(f"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/{model_name}")
model = AutoModelForSequenceClassification.from_pretrained(f"/workspace/Emotion_Intent_Chat/emo_int_chat/intent_reward_model/tuned_model/{model_name}/{checkpoint}/", num_labels=8)

def output_graph(num: int, input_text: str):
    # 入力データの変換と推論
    tokens = tokenizer(input_text, truncation=True, return_tensors='pt')
    tokens = {key: value.to(model.device) for key, value in tokens.items()}
    preds = model(**tokens)

    # 活性化関数
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])

    out_dict = {n: p for n, p in zip(label_names, prob)}

    plt.figure(figsize=(8, 6))
    df = pd.DataFrame(out_dict.items(), columns=['name', 'prob'])
    ax = sns.barplot(x='name', y='prob', data=df)
    plt.title('入力文: ' + input_text, fontsize=15, fontproperties=font_prop)
    
    # 固定されたティック位置を設定し、ラベルを斜めに配置
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='center', fontsize=10, fontproperties=font_prop)
    
    plt.tight_layout()  # レイアウトを調整して、ラベルが重ならないようにする
    plt.savefig(f"Intent_{num}_{input_text}.png")  # グラフを保存する
    plt.close()

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def main():
    model.eval()
    utterances = [
    "これはどうやって使うんですか？",
    "なぜそんなことが起こったんでしょうか？",
    "いつその計画を始める予定ですか？",
    "なるほど、そうなんですね。",
    "おっしゃる通りです。",
    "その点については理解しました。",
    "大変だったね、無理しないで。",
    "心配しないで、きっと大丈夫だよ。",
    "つらい気持ち、わかるよ。",
    "その意見に賛成です。",
    "確かに、そうですね。",
    "同じ考えです。",
    "頑張って、応援しているよ！",
    "きっと成功するから、信じて。",
    "あなたならできるよ！",
    "それは本当に大変だったね。",
    "気持ち、わかるよ。",
    "そんなことがあったなんて、辛かったね。",
    "これを試してみたらどうでしょうか？",
    "もう少し休んだほうがいいと思います。",
    "こうしたらもっと良くなるかもしれません。",
    "良い一日を過ごしてくださいね。",
    "これからもお幸せに。",
    "早く元気になりますように。"
    ]
    for i, input_text in enumerate(utterances):
        output_graph(i, input_text)

if __name__ == "__main__":
    import matplotlib
    print(matplotlib.get_backend())  # 現在のバックエンドを確認
    matplotlib.use('Agg')  # 'Agg'バックエンドを使用
    main()
