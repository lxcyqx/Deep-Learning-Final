from collections import Counter
from getData import getRawData
import pandas as pd
import nltk
import numpy as np

try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# tokenize song lyrics by removing punctuation, non-important words, etc
def clean_lyrics(lyrics):
    lyrics = lyrics.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    lyrics = tokenizer.tokenize(lyrics)
    stemmer = SnowballStemmer('english')
    # removing stopwords (words that do not provide any meaning to the text, such as 'wasn't', 'but', 'will', 'while', 'if')
    stop_words = stopwords.words('english')
    clean_lyrics = [stemmer.stem(word) for word in lyrics if stemmer.stem(word) not in stop_words]
    return clean_lyrics

# get information related to word count from all the lyrics
def preprocess_lyrics():

    rawData = getRawData()
    df = pd.DataFrame(rawData)
    lyrics = df[2]
    word_to_ids = {}
    curr_word_idx = 0
    number_of_songs_word_appears = Counter()
    songs_to_bag_of_words = {}
    labels = {}
    curr_song_idx = 0

    for i in range(len(lyrics)):
        song = clean_lyrics(df[2][i])
        word_count_in_one_song = {} 
        for word in song:
            if word not in word_to_ids:
                word_to_ids[word] = curr_word_idx
                curr_word_idx += 1
            if word in word_count_in_one_song:
                word_count_in_one_song[word] += 1
            else:
                word_count_in_one_song[word] = 1
                number_of_songs_word_appears[word] += 1

        songs_to_bag_of_words[curr_song_idx] = word_count_in_one_song
        labels[curr_song_idx] = df[3][curr_song_idx]
        curr_song_idx += 1

    return songs_to_bag_of_words, number_of_songs_word_appears, word_to_ids, labels

# calculate tf-idf matrix of song_ids by word_ids
def tf_idf(songs, song_word_counts, word_ids):
    number_of_songs = len(songs)
    number_of_words = len(word_ids)
    tf_idf = np.zeros([number_of_songs, number_of_words])
    for song_id, bag_of_words in songs.items():
        for word in bag_of_words:
            total_words = sum(bag_of_words.values())
            tf = float(bag_of_words.get(word) / total_words)
            idf = float(np.log(number_of_songs / song_word_counts.get(word)))
            tf_idf[song_id][word_ids[word]] = float(tf * idf) # matrix [song id] by [word id]
    return tf_idf

# calculate which song has the closest 'norm' by calculating the 
def find_closest_song(song_id, tf_idf_matrix):
    min_dist = float("inf")
    similar_song_id = None
    for i in range(len(tf_idf_matrix)):
        if i != song_id:
            curr_dist = np.linalg.norm(tf_idf_matrix[song_id]-tf_idf_matrix[i])
            if curr_dist < min_dist:
                min_dist = curr_dist
                similar_song_id = i
    return similar_song_id


def main():
    songs, song_word_counts, word_ids, labels = preprocess_lyrics()
    tf_idf_matrix = tf_idf(songs, song_word_counts, word_ids)
    song_ids = np.random.choice(np.arange(831), size=100)
    count = 0
    for i in song_ids:
        similar_song_id = find_closest_song(i, tf_idf_matrix)
        if labels[i] == labels[similar_song_id]:
            count += 1
        print(labels[i], labels[similar_song_id])
    print(count / 100)

if __name__ == '__main__':
    main()