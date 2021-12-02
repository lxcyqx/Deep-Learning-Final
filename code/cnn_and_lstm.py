import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D
from tensorflow.keras.layers import LSTM,Dropout,Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import sys
# fix random seed for reproducibility
np.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 10000
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=top_words)
train_x, valid_x, train_y, valid_y = train_test_split(train_x,train_y,test_size = 0.2)
print("Shape of train data:", train_x.shape)
print(train_x[:20])
print(train_y[:20])
print("Shape of Test data:", test_x.shape)
print("Shape of CV data:", valid_x.shape)

# truncate and pad input sequences
max_review_length = 600
train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)
test_x = sequence.pad_sequences(test_x, maxlen=max_review_length)
valid_x = sequence.pad_sequences(valid_x,maxlen=max_review_length)

def lstm():
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # model.fit(train_x, train_y, epochs=5, batch_size=256,verbose = 1,validation_data=(valid_x,valid_y))
    # model.save("../checkpoints/weights_best.hdf5")

    # Final evaluation of the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights("../checkpoints/weights_best.hdf5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    scores = model.evaluate(test_x, test_y, verbose=1,batch_size = 256)
    print("LSTM Accuracy: %.2f%%" % (scores[1]*100))

    return model

def cnn_and_lstm():
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x, train_y, epochs=5, batch_size=256,verbose = 1,validation_data=(valid_x,valid_y))
    model.save("../checkpoints/weights_lstm_cnn.hdf5")

    # Final evaluation of the model
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.load_weights("../checkpoints/weights_lstm_cnn.hdf5")
    scores = model.evaluate(test_x, test_y, verbose=0)
    print("CNN + LSTM Accuracy: %.2f%%" % (scores[1]*100))

    return model

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"LSTM", "CNN_LSTM"}:
        print("USAGE: python3 cnn_and_lstm.py <Model Type>")
        print("<Model Type>: [LSTM/CNN_LSTM]")
        exit()
    # Initialize model
    if sys.argv[1] == "LSTM":
        model = lstm()
        # phrase = "The movie wasn't all that great. Plot was bad, characters were lackluster. I fell asleep half way through."
        # phrase = sequence.pad_sequences(train_x, maxlen=max_review_length)
    elif sys.argv[1] == "CNN_LSTM":
        model = cnn_lstm()

if __name__ == '__main__':
    main()