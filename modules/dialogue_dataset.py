import pandas as pd
import pickle

# Excelデータの読み込み
df = pd.read_excel("/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/japanese_empathetic_dialogues.xlsx", sheet_name='対話')

# 1ターンの会話、3ターンの会話をそれぞれ抽出
one_turn_index = []
three_turn_index = []
for i in range(80000):
    if i % 4 == 0:
        one_turn_index.append(i)
    if i % 4 != 3:
        three_turn_index.append(i)

df_1 = df.loc[one_turn_index]
df_3 = df.loc[three_turn_index]

print(df_1)
print(df_3)

conversation_lists = []
for df_n in [df_1, df_3]:
    # 'ID'でグループ化して '話者' と '発話' のカラムをまとめてリスト化
    grouped_df = df_n.groupby('ID').apply(lambda x: x[['話者', '発話']].to_dict('records')).reset_index(name='対話')

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
    conversation_lists.append(conversation_list)

conversation_list_full = conversation_lists[0] + conversation_lists[1]
print(conversation_list_full)
# 結果を確認
print("full")
print(conversation_list_full[:5])  # 最初の5要素を確認
print("1 turn")
print(conversation_lists[0][:5])  # 最初の5要素を確認
print("3 turns")
print(conversation_lists[1][:5])  # 最初の5要素を確認

# すべて4件の会話かどうかを確認 -> True
# print("start output")
# for i in conversation_list:
#     if len(i) != 4:
#         print(i)
# print("end output")

# with open('/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue.pkl', 'wb') as f:
#     pickle.dump(conversation_list_full, f)

with open('/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_1turn.pkl', 'wb') as f:
    pickle.dump(conversation_lists[0], f)

with open('/workspace/Emotion_Intent_Chat/JEmpatheticDialogue/JEmpatheticDialogue_3turn.pkl', 'wb') as f:
    pickle.dump(conversation_lists[1], f)
    
print("pkl file was saved.")