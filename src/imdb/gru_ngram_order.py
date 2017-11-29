# -*- coding: utf-8 -*-

'''implementation of bi-gru+n-gram(order) with 3-gram in our paper'''

import pickle
import os
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, SpatialDropout1D, GRU

num_words = 30000
max_len = 400
embedding_dimension = 300
data_dir = '../data'


def get_input():
    with open(os.path.join(data_dir, "imdb_texts.pkl"), mode='rb') as f:
        texts = pickle.load(f)

    tokenizer = Tokenizer(nb_words=num_words)
    tokenizer.fit_on_texts(texts[0:25000])
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_reverse = [list(reversed(seq)) for seq in sequences]

    x = pad_sequences(sequences, maxlen=max_len)
    x_reverse = pad_sequences(sequences_reverse, maxlen=max_len)

    word_index = tokenizer.word_index
    embeddings_index = {}
    wordX = np.load(open(os.path.join(data_dir, "embedding.300d.npy"),mode='rb'))
    allwords = pickle.load(open(os.path.join(data_dir, "words.pkl"),mode='rb'))
    print(len(allwords))
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    x_seq = np.zeros((50000, (max_len - 2) * 3), dtype=np.int)
    for i in range(50000):
        for j in range(max_len - 2):
            x_seq[i, j * 3] = x[i, j]
            x_seq[i, j * 3 + 1] = x[i][j + 1] + num_words
            x_seq[i, j * 3 + 2] = x[i][j + 2] + num_words * 2

    x_train_0 = x[:25000]
    x_train_1 = x_reverse[:25000]
    x_train_2 = x_seq[:25000]
    x_test_0 = x[25000:]
    x_test_1 = x_reverse[25000:]
    x_test_2 = x_seq[25000:]

    result = []

    indice = np.arange(25000)
    np.random.shuffle(indice)
    result.append(x_train_0[indice])
    result.append(x_train_1[indice])
    result.append(x_train_2[indice])
    result.append(x_test_0[indice])
    result.append(x_test_1[indice])
    result.append(x_test_2[indice])
    result.append(y_train[indice])
    result.append(y_test[indice])

    result.append(embedding_matrix)
    return result


def get_model(embedding_matrix):
    input_1 = Input((max_len,))
    embedding_1 = Embedding(num_words, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_1)
    x = SpatialDropout1D(0.25)(embedding_1)
    x = GRU(300,
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='relu')(x)
    # x = Dense(300, activation='relu')(x)

    input_2 = Input((max_len,))
    embedding_2 = Embedding(num_words, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_2)
    y = SpatialDropout1D(0.25)(embedding_2)
    y = GRU(300,
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='relu')(y)
    # y = Dense(300, activation='relu')(y)

    input_3 = Input(((max_len - 2) * 3,))
    embedding_3 = Embedding(num_words * 3, embedding_dimension)(input_3)
    z = AveragePooling1D(pool_size=3, strides=3, padding='valid')(embedding_3)
    z = GlobalMaxPooling1D()(z)

    a = keras.layers.concatenate([x, y, z])
    output_1 = Dense(1, activation='sigmoid')(a)

    model = Model(inputs=[input_1, input_2, input_3], outputs=[output_1])
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    result = get_input()
    model = get_model(result[8])
    model.fit(result[0:3], result[6],
              batch_size=128,
              nb_epoch=50,
              verbose=2,
              validation_data=(result[3:6], result[7]))
