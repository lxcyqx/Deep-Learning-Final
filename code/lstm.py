import tensorflow as tf

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D, Embedding, Dense, LSTM, GRU, Dropout
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

lyricss = []
labels = []

with open("../preprocessing/songsToMeaningAndLyrics.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[7])
        lyrics = row[3]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            lyrics = lyrics.replace(token, ' ')
            lyrics = lyrics.replace(' ', ' ')
        lyricss.append(lyrics)
train_size = int(len(lyricss) * training_portion)

lyricss, labels = shuffle(lyricss, labels)
train_lyricss = lyricss[0: train_size]
train_labels = labels[0: train_size]

validation_lyricss = lyricss[train_size:]
validation_labels = labels[train_size:]

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(filters=filters, num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_lyricss)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_lyricss)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_lyricss)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

EMBEDDING_DIMENSION = 64

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIMENSION))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(6, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

def getLyrics(title):
    with open("../preprocessing/songsToMeaningAndLyrics.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            song_name = row[2]
            if song_name == title:
                return row[3]

def predict(song_name):
    lyrics = getLyrics(song_name)
    txt = [lyrics]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    print(song_name, pred, labels[np.argmax(pred)])

def main():
    if len(sys.argv) != 2:
        print("USAGE: python3 lstm.py \"<song name>\"")
        exit()
    
    song_name = sys.argv[1].lower()
    predict(song_name)

if __name__ == '__main__':
    main()