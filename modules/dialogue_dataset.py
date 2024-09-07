import pandas as pd
import pickle

# Excelデータの読み込み
df = pd.read_excel("/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/japanese_empathetic_dialogues.xlsx", sheet_name='対話')

# 'ID'でグループ化して '話者' と '発話' のカラムをまとめてリスト化
grouped_df = df.groupby('ID').apply(lambda x: x[['話者', '発話']].to_dict('records')).reset_index(name='対話')

# '対話'カラムの各レコードに対して処理を適用
def replace_speaker_and_column_name(dialogues):
    # 各辞書のカラム名を変更し、話者を user/assistant に置き換え
    for dialogue in dialogues:
        dialogue['role'] = 'user' if dialogue['話者'] == 'A' else 'assistant'
        dialogue['content'] = dialogue.pop('発話')
        dialogue.pop('話者')  # '話者' カラムはもう不要なので削除
    return dialogues

# 各対話に対して適用
grouped_df['対話'] = grouped_df['対話'].apply(replace_speaker_and_column_name)

# '対話'カラムの内容をリスト化
conversation_list = grouped_df['対話'].tolist()

# 結果を確認
# print(conversation_list[:5])  # 最初の5要素を確認

# すべて4件の会話かどうかを確認 -> True
# print("start output")
# for i in conversation_list:
#     if len(i) != 4:
#         print(i)
# print("end output")

with open('/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue.pkl', 'wb') as f:
    pickle.dump(conversation_list, f)
    
print("pkl file was saved.")