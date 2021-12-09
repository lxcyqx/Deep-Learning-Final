import nlpaug
import nlpaug.augmenter.word as naw
import nltk
import csv
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

aug = naw.SynonymAug(aug_src='wordnet',aug_max=7)

write_to = open('../data/songsToMeaningAndLyricsFull.csv', 'w')

with open("../data/songsToMeaningAndLyricsReal.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        words = row[3]
        augmented_res = aug.augment(words, n=3)
        for augmented in augmented_res:
            row[3] = augmented
            with open('../data/songsToMeaningAndLyricsFull.csv', 'a') as output:
                writer = csv.writer(output)
                writer.writerow(row)