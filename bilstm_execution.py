from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from janome.tokenizer import Tokenizer  # 追加
import numpy as np
import random
import sys
import io

import keras
import numpy

path = './data_clear.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

#chars = sorted(list(set(text)))
#print('total chars:', len(chars))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

text =Tokenizer().tokenize(text, wakati=True)  # 分かち書きする
chars = text
count = 0
char_indices = {}  # 辞書初期化
indices_char = {}  # 逆引き辞書初期化

for word in chars:
    if not word in char_indices:  # 未登録なら
       char_indices[word] = count  # 登録する
       count +=1
       #print(count,word)  # 登録した単語を表示
# 逆引き辞書を辞書から作成する
indices_char = dict([(value, key) for (key, value) in char_indices.items()])

print(len(chars))

'''
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 5
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
#print('nb sequences:', len(sentences))

#print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1 #sentencesのi番目の配列のt番目の文字は辞書のchar番目
    y[i, char_indices[next_chars[i]]] = 1#sentencesのi番目の配列の次につながる文字
'''

model = keras.models.load_model("bilstm_leng10.h5")

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
practice = []
practice2 = []

def on_epoch_end():
   # Function invoked at end of each epoch. Prints generated text.
   #print()
   #print('----- Generating text after Epoch: %d' % epoch)

   #start_index = random.randint(0, len(text) - maxlen - 1)
   start_index = 0  # テキストの最初からスタート
   for diversity in [0.1]:  # diversity は 0.2のみ使用
       #print('----- diversity:', diversity)

       generated = ''
       sentence = text[start_index: start_index + maxlen]
       test = text[start_index: start_index + maxlen]
       # sentence はリストなので文字列へ変換して使用
       generated += "".join(sentence)
       #print(sentence)
       
       # sentence はリストなので文字列へ変換して使用
       #print('----- Generating with seed: "' + "".join(sentence)+ '"')
       #sys.stdout.write(generated)#ここがprintだと行がずれてしまう

       for i in range(len(text)-maxlen):
           x_pred = np.zeros((1, maxlen, len(chars)))
           for t, char in enumerate(sentence):
               x_pred[0, t, char_indices[char]] = 1.

           preds = model.predict(x_pred, verbose=0)[0]
           next_index = sample(preds, diversity)
           next_char = indices_char[next_index]

           generated += next_char
           sentence = sentence[1:]
           # sentence はリストなので append で結合する
           sentence.append(next_char)
           test.append(next_char)

           #sys.stdout.write(next_char)#ここをgeneratedにするとどんどん新しい文がつながって描かれてしまう
           #sys.stdout.flush()
       #print()
       #print(test)
       global practice
       practice = test[:]

def on_epoch_end2():
       # Function invoked at end of each epoch. Prints generated text.
       #print()
       #print('----- Generating text after Epoch: %d' % epoch)

       #start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0  # テキストの最初からスタート
    for diversity in [0.2]:  # diversity は 0.2のみ使用
           #print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        test2 = text[start_index: start_index + maxlen]
        # sentence はリストなので文字列へ変換して使用
        generated += "".join(sentence)
        #print(sentence)
           
        # sentence はリストなので文字列へ変換して使用
        #print('----- Generating with seed: "' + "".join(sentence)+ '"')
        #sys.stdout.write(generated)#ここがprintだと行がずれてしまう

        for i in range(len(text)-maxlen):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentences[i]):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            
            generated += next_char
            #sentence = sentence[1:]
            # sentence はリストなので append で結合する
            sentence.append(next_char)
            test2.append(next_char)
            
        global practice2
        practice2 = test2[:]

       
#print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


#on_epoch_end()
#print(practice)


path = './test1.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
text =Tokenizer().tokenize(text, wakati=True)  # 分かち書きする
maxlen = 10
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
on_epoch_end()
#print(practice)
#print(text)
#print(len(practice))
#print(len(text)
print("-----------入力された文字列")
text_print = ""
text_print += "".join(text)
print(text_print)
print("-----------生成した文字列")
practice_print = ""
practice_print += "".join(practice)
print(practice_print)

x = np.zeros((len(practice), len(chars)), dtype=np.bool)
y = np.zeros((len(text), len(chars)), dtype=np.bool)
for i, char in enumerate(practice):
    x[i, char_indices[char]] = 1

for i, char in enumerate(text):
    y[i, char_indices[char]] = 1

#print(x)
#print(y)

result = []
final = ''

for i in range(len(text)):
    check = (x[i] == y[i]).all()
    result.append(check)
    if(result[i] == False):
        final += "".join('\033[31m'+ text[i] +'\033[0m')
        final += "".join("(" + '\033[31m'+ practice[i] +'\033[0m' + ")")
    else:
        final += "".join(text[i])

print("-----------添削例１")
print(final)

on_epoch_end2()
#print(practice2)
print("-----------生成した文字列2")
practice2_print = ""
practice2_print += "".join(practice2)
print(practice2_print)

x = np.zeros((len(practice2), len(chars)), dtype=np.bool)
y = np.zeros((len(text), len(chars)), dtype=np.bool)
for i, char in enumerate(practice2):
    x[i, char_indices[char]] = 1

for i, char in enumerate(text):
    y[i, char_indices[char]] = 1

#print(x)
#print(y)

result2 = []
final2 = ''

for i in range(len(text)):
    check = (x[i] == y[i]).all()
    result2.append(check)
    if(result2[i] == False):
        final2 += "".join('\033[31m'+ text[i] +'\033[0m')
        final2 += "".join("(" + '\033[31m'+ practice2[i] +'\033[0m' + ")")
    else:
        final2 += "".join(text[i])

print("-----------添削例2")
print(final2)
