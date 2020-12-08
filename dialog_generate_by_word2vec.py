from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import datetime
import openpyxl
import glob
import os
import re
import pprint
import math
import numpy as np
import MeCab
import gensim
import string
import random
import sys
import io

# from keras.models import load_weights
from keras.models import load_model
from keras.models import model_from_json

global graph

# clear_session()

# 初期化処理とモデルの読み込み
path = "./dialog(1file).txt"
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
# print('corpus length:', len(text))
text = re.sub(r'「|」|『|』','',text)
text = re.sub(r'人名|施設|地名|場所|','',text)
text = re.sub(r'。|,|、|#|ー','',text)
print(text)

# 形態素で単語単位に分割
tagger = MeCab.Tagger('-Owakati')

# 単語をインデックスに変換
text = tagger.parse(text).split(' ')

chars = text
count = 0
char_indices = {}  # 辞書初期化
indices_char = {}  # 逆引き辞書初期化
 
for word in chars:
    if not word in char_indices:  # 未登録なら
       char_indices[word] = count  # 登録する      
       count +=1
       print(count,word)  # 登録した単語を表示

# 逆引き辞書を辞書から作成する
indices_char = dict([(value, key) for (key, value) in char_indices.items()])

maxlen = 5
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))



print('\nPreparing the data for LSTM...')
# train_x = np.zeros([len(sentences), maxlen], dtype=np.int8)
# train_y = np.zeros([len(sentences)], dtype=np.int8)
train_x = np.abs(np.zeros([len(sentences), maxlen], dtype=np.uint8))
train_y = np.abs(np.zeros([len(sentences)], dtype=np.uint8))
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)



batch_size = 32

# コーパスが巨大でありMemmoryErrorが発生するため、ジェネレータでミニバッチにわけて訓練する
def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = shuffled_data[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()

train_steps, train_batches = batch_iter(train_x, train_y, batch_size)

# 確率的に次の単語を選ぶ関数
def sample(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    dist = np.exp(preds)/np.sum(np.exp(preds))
    choices = range(len(preds))
    return np.random.choice(choices, p=dist)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

# modelの読み込み
# model = model_from_json(json.dumps("dialog_model.json"))

with open("dialog_model_by_word2vec.json", "r") as json_file:
    model = model_from_json(json_file.read())
# model = load_weights("dialog_model.h5", compile=False)
model = load_model("dialog_model_by_word2vec.h5")
model.summary()
graph = tf.get_default_graph()


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
