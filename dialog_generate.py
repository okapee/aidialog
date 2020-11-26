import random
import sys
import io
import json
from keras.engine.saving import load_weights_from_hdf5_group
import numpy as np
import tensorflow as tf

# from keras.models import load_weights
from keras.backend import clear_session
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


global graph

# clear_session()

# 初期化処理とモデルの読み込み
path = "./dialog.txt"
with io.open(path) as f:
    text = f.read().lower()
print("corpus length:", len(text))

chars = sorted(list(set(text)))
print("total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# maxlenずつ次の１語を予測
# stepずつ時間軸をスライドしていく
maxlen = 8
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("nb sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# modelの読み込み
# model = model_from_json(json.dumps("dialog_model.json"))

with open("dialog_model.json", "r") as json_file:
    model = model_from_json(json_file.read())
# model = load_weights("dialog_model.h5", compile=False)
model = load_model("dialog_model.h5")
model.summary()
graph = tf.get_default_graph()

# model = open("dialog_model.json").read()
# model = model_from_json(model)
# # model = model_from_json(model)
# # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# model.load_weights("dialog_model.h5")
# optimizer = RMSprop(lr=0.01)
# model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def generate():
    generated = ""
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0  # 毎回、文章の最初から文章生成
    for diversity in [0.2]:  # diversityが大きくなるにつれ、確率分布から確率の低い文字をサンプリングするようになる
        print("----- diversity:", diversity)

        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            with graph.as_default():
                preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

    return generated


def sample(preds, temperature=1.0):
    # 確率を格納した配列からサンプリングする
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
