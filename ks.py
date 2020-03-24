#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:15:55 2020

@author: seangao
"""

import pandas as pd
from datetime import datetime
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import gensim
import contractions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Input, Embedding, LSTM, GlobalMaxPool1D, Dense, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping

#LOAD DATA
datapath = '/Users/seangao/Desktop/Research/kickstarter/kickstarter-projects/ks-projects-201801.csv'

ks = pd.read_csv(datapath)

list(ks.columns)

#PREPARE DAYS COLUMN
def getDays(launch_date, deadline):
    time1 = datetime.strptime(launch_date, '%Y-%m-%d %H:%M:%S')
    date1 = time1.date()
    
    time2 = datetime.strptime(deadline, '%Y-%m-%d')
    date2 = time2.date()
    
    delta = date2 - date1
    days = delta.days
    return days

ks['days'] = ks.apply(lambda x: getDays(x['launched'], x['deadline']), axis=1)
    
#CONVERT OTHER COUNTRIES
def cvtctry(country):
    if country not in ('US', 'GB', 'CA'):
        country = 'OTHER'
        return country
    else:
        country = country
        return country       
    
ks['country'] = ks['country'].apply(lambda x: cvtctry(x))

ks = pd.get_dummies(ks, columns=['main_category'])
ks = pd.get_dummies(ks, columns=['country'])

#REMOVE UNDEFINED AND LIVE CASES
ks.drop(ks[ks['state'] == 'undefined'].index, inplace=True)
ks.drop(ks[ks['state'] == 'live'].index, inplace=True)

def sttlb(state):
    if state == 'successful':
        return 1
    else:
        return 0

ks['label'] = ks['state'].apply(lambda x: sttlb(x))

#PROPROCESS TEXT
ks['name'] = ks['name'].astype(str)
ks['category'] = ks['category'].astype(str)
ks['text'] = ks[['name','category']].agg(' '.join, axis=1) 

ks.iloc[0]['text']

text_lines = list()
lines = ks['text'].values.tolist()

for line in tqdm(lines):
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stpw = set(stopwords.words('english'))
    words = [w for w in words if not w in stpw]
    text_lines.append(words)

#WORD2VEC
model = gensim.models.Word2Vec(sentences=text_lines,
                               size=128,
                               window=5,
                               workers=4,
                               min_count=1)

words = list(model.wv.vocab)

savepath = '/Users/seangao/Desktop/Research/kickstarter/ks_embedding.txt'
model.wv.save_word2vec_format(savepath, binary=False)

#PREPARE TWO INPUTS
Xt = pd.DataFrame(columns=['text'])
Xt['text'] = ks['text']

Xt['text'] = Xt['text'].apply(lambda x: contractions.fix(x))
Xt['text'] = Xt['text'].apply(lambda x: x.lower())
Xt['text'] = Xt['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
Xt['text'] = Xt['text'].apply(lambda x: x.replace('  ', ' ')) #FIX DOUBLE SPACES

def rmvstpw(input_str):
    str_words = input_str.split()
    keep_words = [word for word in str_words if word not in stpw]
    output_str = ' '.join(keep_words)
    return output_str

Xt['text'] = Xt['text'].apply(lambda x: rmvstpw(x))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(Xt['text'].values)
Xt = tokenizer.texts_to_sequences(Xt['text'].values)
Xt = pad_sequences(Xt)

list(ks.columns)

Xn = ks[['usd_goal_real',
 'days',
 'main_category_Art',
 'main_category_Comics',
 'main_category_Crafts',
 'main_category_Dance',
 'main_category_Design',
 'main_category_Fashion',
 'main_category_Film & Video',
 'main_category_Food',
 'main_category_Games',
 'main_category_Journalism',
 'main_category_Music',
 'main_category_Photography',
 'main_category_Publishing',
 'main_category_Technology',
 'main_category_Theater',
 'country_CA',
 'country_GB',
 'country_OTHER',
 'country_US']]

scaler = MinMaxScaler()
Xn[['usd_goal_real']] = scaler.fit_transform(Xn[['usd_goal_real']])
Xn[['days']] = scaler.fit_transform(Xn[['days']])

y = ks['label'].to_list()

Xt_train, Xt_test, y_train, y_test = train_test_split(
        Xt, y, test_size=0.2, random_state=42)

Xn_train, Xn_test, y_train, y_test = train_test_split(
        Xn, y, test_size=0.2, random_state=42)

#LOAD EMBEDDING
embeddings_index = {}
f = open(savepath)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 128))
for word, i in word_index.items():
    if i > len(word_index) + 1:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#PREPARE MODEL
input_1 = Input(shape=(Xt.shape[1],))
input_2 = Input(shape=(Xn.shape[1],))

embedding_layer = Embedding(len(word_index) + 1, 128, weights=[embedding_matrix])(input_1)
lstm_layer = LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(embedding_layer)
globalmaxpool_layer = GlobalMaxPool1D()(lstm_layer)

dense_layer_1 = Dense(32, activation='relu')(input_2)
dropout_layer_1 = Dropout(0.2)(dense_layer_1)
dense_layer_2 = Dense(32, activation='relu')(dropout_layer_1)
dropout_layer_2 = Dropout(0.2)(dense_layer_2)

concat_layer = Concatenate()([globalmaxpool_layer, dropout_layer_2])
dense_layer_3 = Dense(64, activation='relu')(concat_layer)
dropout_layer_3 = Dropout(0.2)(dense_layer_3)

output = Dense(1, activation='sigmoid')(dense_layer_3)

model = Model(inputs=[input_1, input_2], outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

#TRAIN AND TEST
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit([Xt_train, Xn_train], y_train,
                    epochs=100, batch_size=1024, verbose=1,
                    validation_data=([Xt_test, Xn_test], y_test),
                    callbacks=[es])