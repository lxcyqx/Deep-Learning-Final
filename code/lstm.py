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

articles = []
labels = []

with open("../preprocessing/songsToMeaningAndLyrics.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[7])
        article = row[3]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)
train_size = int(len(articles) * training_portion)

articles, labels = shuffle(articles, labels)
train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# model = tf.keras.Sequential([
#     # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
#     tf.keras.layers.Embedding(vocab_size, embedding_dim),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
# #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     # use ReLU in place of tanh function since they are very good alternatives of each other.
#     tf.keras.layers.Dense(embedding_dim, activation='relu'),
#     # Add a Dense layer with 6 units and softmax activation.
#     # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
#     tf.keras.layers.Dense(5, activation='softmax')
# ])

# model.summary()

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

num_epochs = 20
print("training label seq ", training_label_seq )
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

# should be love
txt = ["come on skinny love just last the year, pour a little salt we were never here, my my my, my my my, my-my-my my-my, staring at the sink of blood and crushed veneer, tell my love to wreck it all, cut out all the ropes and let me fall, my my my, my my my, my-my-my my-my, right in the moment this order's tall, and i told you to be patient, and i told you to be fine, and i told you to be balanced, and i told you to be kind, and in the morning i'll be with you, but it will be a different kind, 'cause i'll be holding all the tickets, and you'll be owning all the fines, come on skinny love, what happened here?, suckle on the hope in light brassieres, my my my, my my my, my-my-my my-my, sullen load is full, so slow on the split, and i told you to be patient, and i told you to be fine, and i told you to be balanced, and i told you to be kind, and now all your love is wasted, then who the hell was i?, 'cause now i'm breaking at the britches, and at the end of all your lines, who will love you?, who will fight?, and who will fall far behind?, come on skinny love, my my my, my my my, my-my-my my-my, my my my, my my my, my-my-my my-my"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
labels = ['love', 'party', 'breakup', 'sex', 'religion']
print(pred)
print(np.argmax(pred))
print(pred, labels[np.argmax(pred)])

# should be break up
txt = ["you told me, there's no need, to talk it out, 'cause it's too late, to proceed, and slowly, i took your words, and walked away, no looking back, i won't regret, no, i will find my way, i'm broken, but still i have to say, it's alright, ok, i'm so much better without you, i won't be sorry, alright, ok, so don't you bother what i do, no matter what you say, i won't return, our bridge has burned down, i'm stronger now, alright, ok, i'm so much better without you, i won't be sorry, you played me, betrayed me, your love was nothing but a game, portrayed a role, you took control, i, i couldn't help but fall, so deep, but now i see things clear, it's alright, ok, i'm so much better without you, i won't be sorry, alright, ok, so don't you bother what i do, no matter what you say, i won't return, our bridge has burned down, i'm stronger now, alright, ok, i'm so much better without you, i won't be sorry, don't waste your fiction tears on me, just save them for someone in need, it's way too late, i'm closing the door, it's alright, ok, i'm so much better without you, i won't be sorry, alright, ok, so don't you bother what i do, no matter what you say, i won't return, our bridge has burned down, i'm stronger now, alright, ok, i'm so much better without you, i won't be sorry, it's alright, ok, alright, ok, without you, no matter what you say, it's alright, ok, alright, ok, without you, i won't be sorry"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(pred)
print(np.argmax(pred))
print(pred, labels[np.argmax(pred)])

# should be religion
txt = ["i don't know about tomorrow, i just live from day to day, i don't borrow from the sunshine, for the skies they turn to grey. and i don't worry for the future, for i know what jesus said, and today i'll walk beside him, for he's what lies ahead. many things about tomorrow, i don't seem to understand, but i know who holds tomorrow, and i know who holds my hand. every step is getting brighter, as the golden stairs i climb, every burden's getting lighter, every cloud is silver-lined. there the sun is always shining, there no tear will ever dim the eye, at the ending of the rainbow, where the mountains touch the sky. many things about tomorrow, i don't seem to understand, but i know who holds tomorrow, and i know who holds my hand."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(pred)
print(np.argmax(pred))
print(pred, labels[np.argmax(pred)])

# should be party
txt = ["as he came into the window, was a sound of a crescendo, he came into her apartment, he left the bloodstains on the carpet, she was sitting at the table, he could see she was unable, so she ran into the bedroom, she was struck down, it was her doom, annie, are you ok, are you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, will you tell us that you're ok, there's a sign at the window, that he struck you, a crescendo, annie, he came into your apartment, he left the bloodstains on the carpet, then you ran into the bedroom, you were struck down, it was your doom, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, you've been hit by, you've been struck by, a smooth criminal, so they came into the outway, it was sunday, what a black day, i could made a salutation, sounding heartbeats, intimidations, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, will you tell us that you're ok, there's a sign at the window, that he struck you, a crescendo, annie, he came into your apartment, he left the bloodstains on the carpet, then you ran into the bedroom, you were struck down, it was your doom, annie, are you ok, you ok, are you ok, annie, you've been hit by, you've been struck by, a smooth criminal, annie, are you ok, will you tell us that you're ok, there's a sign at the window, that he struck you, a crescendo, annie, he came into your apartment, he left the bloodstains on the carpet, then you ran into the bedroom, you were struck down, it was your doom, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie, annie, are you ok, you ok, are you ok, annie"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(pred)
print(np.argmax(pred))
print(pred, labels[np.argmax(pred)])

def main():
    if len(sys.argv) != 2:
        print("USAGE: python3 lstm.py \"<song name>\"")
        exit()
    
    song_name = sys.argv[1]
    print("SONG -------------------", song_name)

if __name__ == '__main__':
    main()