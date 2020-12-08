import MeCab
import random
import sys
import io
import re
import numpy as np
import tensorflow as tf

# from keras.models import load_weights
from keras.models import load_model
from keras.models import model_from_json

global graph

# clear_session()

# 初期化処理とモデルの読み込み
# path = "./dialog(1file).txt"
path = "./dialog_mini.txt"
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
print("corpus length:", len(text))

text = re.sub(r"「|」|『|』", "", text)
text = re.sub(r"人名|施設|地名|場所|", "", text)
text = re.sub(r"。|,|、|#|ー", "", text)

# 形態素で単語単位に分割
tagger = MeCab.Tagger("-Owakati")

# 形態素で単語単位に分割
tagger = MeCab.Tagger("-Owakati")

# 単語をインデックスに変換
text = tagger.parse(text).split(" ")

chars = text
count = 0
char_indices = {}  # 辞書初期化
indices_char = {}  # 逆引き辞書初期化

for word in chars:
    if not word in char_indices:  # 未登録なら
        char_indices[word] = count  # 登録する
        count += 1
        print(count, word)  # 登録した単語を表示
# 逆引き辞書を辞書から作成する
indices_char = dict([(value, key) for (key, value) in char_indices.items()])

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 5
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("nb sequences:", len(sentences))


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.uint8)
y = np.zeros((len(sentences), len(chars)), dtype=np.uint8)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# modelの読み込み

# with open("dialog_model.json", "r") as json_file:
#     model = model_from_json(json_file.read())
with open("dialog_model_by_word_onehot.json", "r") as json_file:
    model = model_from_json(json_file.read())

model = load_model("dialog_model_by_word_onehot.h5")
model.summary()
graph = tf.get_default_graph()


def generate():
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0  # テキストの最初からスタート
    for diversity in [0.2]:  # diversity は 0.2のみ使用
        print("----- diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        # sentence はリストなので文字列へ変換して使用
        generated += "".join(sentence)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            # sentence はリストなので append で結合する
            sentence.append(next_char)

    return generated


def sample(preds, temperature=1.0):
    # 確率を格納した配列からサンプリングする
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

