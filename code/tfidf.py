from collections import Counter
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import numpy as np
import sys
import requests

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#GENIUS API STUFF
GENIUS_API_TOKEN='YcO4NV1shV8O0pA4kw9eCJflks0JpecSoMpU2v8sc2fitjDfKfHpGOClBJSAzstM'

TITLE_IDX = 2
LYRICS_IDX = 3
LABEL_IDX = 4

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
def preprocess_lyrics(data):
    df = pd.DataFrame(data)
    lyrics = df[LYRICS_IDX]
    word_to_ids = {}
    curr_word_idx = 0
    number_of_songs_word_appears = Counter()
    songs_to_bag_of_words = {}
    labels = {}
    curr_song_idx = 0
    id_to_name = {}
    name_to_id = {}

    for i in range(len(lyrics)):
        song = clean_lyrics(df[LYRICS_IDX][i])
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
        labels[curr_song_idx] = df[LABEL_IDX][curr_song_idx]
        id_to_name[curr_song_idx] = df[TITLE_IDX][i] # first idx?
        name_to_id[df[TITLE_IDX][i]] = curr_song_idx
        curr_song_idx += 1
    return songs_to_bag_of_words, number_of_songs_word_appears, word_to_ids, labels, id_to_name, name_to_id

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
            if curr_dist <= min_dist:
                min_dist = curr_dist
                similar_song_id = i
    return similar_song_id, min_dist


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
    replaceArtist = artist.replace(" ","-")
    replaceName = song.replace(" ", "-")
    newStr = initialStr + replaceArtist + "-" + replaceName + "-lyrics"
    return scrape_song_lyrics(newStr)

# If a song name is given, it returns the closest song to it; otherwise it predicts randomly
def main():
    dataNp = pd.read_csv('../data/songsToMeaningAndLyricsReal.csv')
    # removing lyrics with less than 20 words and more than 100 words
    drop_rows=[72, 81, 132, 133, 427, 429, 496, 530, 551, 580, 583, 684, 705, 721, 921, 926, 930, 934, 955, 1015, 1063, 1075, 1144, 1169, 1193, 1205, 1209, 1296, 1301, 1334, 1338, 1357, 1365, 1376, 1415, 1417, 1431, 1482, 1488, 1490, 1493, 1498, 1500, 1501, 1515, 1541, 1549, 1576, 1577, 1589, 1591, 1601, 1679, 1684, 1701, 1706, 1735, 1763, 1770, 1796, 1840, 1846, 1847, 1864, 1882, 1925, 1934, 1939, 1944, 1955, 1957, 2020, 2026, 2054, 2096, 2105, 2108]
    rows = dataNp.index[drop_rows]

    dataNp.drop(rows, inplace=True)
    dataNp = np.array(dataNp)

    for i in range(len(dataNp)):
        dataNp[i][0] = i

    songs, song_word_counts, word_ids, labels, id_to_name, name_to_id = preprocess_lyrics(dataNp)
    if len(sys.argv) == 2:
        song_name = sys.argv[1]
        if song_name in name_to_id:
            song_ids = [name_to_id[song_name]]
        else: 
            print("could not find song")
            exit()
    else:
        song_ids = np.random.choice(np.arange(2102), size=10)
    tf_idf_matrix = tf_idf(songs, song_word_counts, word_ids)
    count = 0
    control = ["breakup", "love"]
    control_count = 0
    for i in song_ids:
        similar_song_id, norm = find_closest_song(i, tf_idf_matrix)
        if labels[i] == labels[similar_song_id]:
            count += 1
        elif labels[i] in control and labels[similar_song_id] in control:
            control_count += 1
        print("input: ", id_to_name[i], " recommendation: ", id_to_name[similar_song_id])
        print("input label: ", labels[i], " recommendation label: ", labels[similar_song_id])

    print("accuracy: ", count / len(song_ids))
    print("% breakup and love were mixed: ", control_count/ len(song_ids))

if __name__ == '__main__':
    main()