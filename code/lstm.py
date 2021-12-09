import tensorflow as tf

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
import requests
# Scrape data from an HTML document
from bs4 import BeautifulSoup
# I/O
import os
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D, Embedding, Dense, LSTM, GRU, Dropout
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#GENIUS API STUFF
GENIUS_API_TOKEN='YcO4NV1shV8O0pA4kw9eCJflks0JpecSoMpU2v8sc2fitjDfKfHpGOClBJSAzstM'

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

lyricss = []
labels = []

with open("../data/songsToMeaningAndLyricsFull.csv", 'r') as csvfile:
    print("in open file")
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[4])
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

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels)) - 1
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels)) - 1

print(train_labels[:10])
print(training_label_seq[:10])

def lstm():
    EMBEDDING_DIMENSION = 32
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIMENSION))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_lstm():
    EMBEDDING_DIMENSION = 32
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIMENSION))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def bidirectional_lstm():
    EMBEDDING_DIMENSION = 64

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIMENSION))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

def cnn_bidirectional_lstm():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Conv1D(filters=24, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=False)))
    model.add(Dense(5,activation='softmax'))

def train():
    model1 = lstm()
    model2 = cnn_lstm()
    model3 = bidirectional_lstm()
    model4 = cnn_bidirectional_lstm()

    num_epochs = 20
    history = model1.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
    history2 = model2.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
    history3 = model3.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
    history4 = model4.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

    model1.save("../checkpoints/lstm")
    model2.save("../checkpoints/cnn_lstm")
    model3.save("../checkpoints/bidirectional_lstm")
    model4.save("../checkpoints/cnn_bidirectional_lstm")

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
# plot_graphs(history, "LSTM accuracy")
# plot_graphs(history, "LSTM loss")

# plot_graphs(history2, "CNN + LSTM accuracy")
# plot_graphs(history2, "CNN + LSTM loss")

# plot_graphs(history3, "Bidirectional LSTM accuracy")
# plot_graphs(history3, "Bidirectional LSTM loss")

# plot_graphs(history4, "CNN Bidirectional LSTM accuracy")
# plot_graphs(history4, "CNN Bidirectional LSTM loss")
model1 = tf.keras.models.load_model("../checkpoints/lstm")
model2 = tf.keras.models.load_model("../checkpoints/cnn-lstm")
model3 = tf.keras.models.load_model("../checkpoints/bidirectional-lstm")
model4 = tf.keras.models.load_model("../checkpoints/cnn-bidirectional-lstm")

def scrape_song_lyrics(url):

    try:
        page = requests.get(url)
    except requests.exceptions.RequestException as e:  # This is the correct syntax

        return None 
    html = BeautifulSoup(page.text, 'html.parser')
    classToDig = html.find("div",class_="Lyrics__Container-sc-1ynbvzw-6 lgZgEN")
    if classToDig is None:
        return None
    else:
        lyrics = classToDig.get_text()
    #print(url + ' survived!')
    #remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
    #remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    lyrics = re.sub(r'[^\w]', ' ', lyrics) 
    words = lyrics.split(" ")
    allWords = []
    for word in words:
        res_list = [s for s in re.split("([A-Z][^A-Z]*)", word) if s]
        allWords += res_list
    return " ".join(allWords)

def scrape_lyrics_song(artist, song):
    initialStr = "https://genius.com/"
    # consider removing punctation from artist name and song name 
    replaceArtist = artist.replace(" ","-")
    replaceName = song.replace(" ", "-")
    newStr = initialStr + replaceArtist + "-" + replaceName + "-lyrics"
    return scrape_song_lyrics(newStr)

def getLyrics(title):
    with open("../data/songsToMeaningAndLyricsFull.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            song_name = row[2]
            if song_name == title:
                return row[3]

def predict(lyrics, song_name, artist_name):
    txt = [lyrics]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model1.predict(padded)
    pred2 = model2.predict(padded)
    pred3 = model3.predict(padded)
    pred4 = model4.predict(padded)
    meaning = ["love", "breakup", "party", "sex", "religion"]
    print(song_name, " by ", artist_name, " - meaning: ", meaning[np.argmax(pred)])
    print(song_name, " by ", artist_name, " - meaning: ", meaning[np.argmax(pred2)])
    print(song_name, " by ", artist_name, " - meaning: ", meaning[np.argmax(pred3)])
    print(song_name, " by ", artist_name, " - meaning: ", meaning[np.argmax(pred3)])

def main():
    if len(sys.argv) != 3:
        print("hello")
        print("USAGE: python3 lstm.py \"<song name>\" \"artist name\"")
        exit()
    
    song_name = sys.argv[1].lower()
    artist_name = sys.argv[2].lower()
    lyrics = scrape_lyrics_song(artist_name, song_name)
    if (lyrics == None):
        print("ERROR: could not find song")
    else:
        print(lyrics)
        predict(lyrics, song_name, artist_name)

if __name__ == '__main__':
    main()