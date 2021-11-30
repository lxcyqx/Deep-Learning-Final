from getData import getRawData
import pandas as pd
import nltk

try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

def tfidf():
    rawData = getRawData()
    df = pd.DataFrame(rawData)
    lyrics = df[2]
    mappings_id_lyrics = {}
    seen_words = {}
    curr_word_id = 0
    for i in range(len(lyrics)):
        mappings_id_lyrics[i] = clear_lyrics(df[2][i])

def clear_lyrics(lyrics):
    lyrics = lyrics.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    lyrics = tokenizer.tokenize(lyrics)
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    clean_lyrics = [stemmer.stem(word) for word in lyrics if stemmer.stem(word) not in stop_words]
    return clean_lyrics

def calculate_freq_matrix(cleaned_lyrics, seen_words, curr_word_id):
    freq_matrix = {}
    for word in cleaned_lyrics:
        if word not in seen_words:
            seen_words[word] = curr_word_id
            curr_word_id += 1

        if word in freq_matrix:
            freq_matrix[word] += 1
        else:
            freq_matrix[word] = 1
    return freq_matrix

def calculate_term_freq_matrix(freq_matrix):
    pass

def main():
    tfidf()

if __name__ == '__main__':
    main()