import pandas as pd
import numpy as np
import math

azLyricsNumbersCsv = '../artistSongsLyricsData/azlyrics_lyrics_19.csv'
azLyricsACsv = '../artistSongsLyricsData/azlyrics_lyrics_a.csv'
azLyricsBCsv = '../artistSongsLyricsData/azlyrics_lyrics_b.csv'
azLyricsCCsv = '../artistSongsLyricsData/azlyrics_lyrics_c.csv'
azLyricsDCsv = '../artistSongsLyricsData/azlyrics_lyrics_d.csv'
azLyricsECsv = '../artistSongsLyricsData/azlyrics_lyrics_e.csv'
azLyricsFCsv = '../artistSongsLyricsData/azlyrics_lyrics_f.csv'
azLyricsGCsv = '../artistSongsLyricsData/azlyrics_lyrics_g.csv'
azLyricsHCsv = '../artistSongsLyricsData/azlyrics_lyrics_h.csv'
azLyricsICsv = '../artistSongsLyricsData/azlyrics_lyrics_i.csv'
azLyricsJCsv = '../artistSongsLyricsData/azlyrics_lyrics_j.csv'
azLyricsKCsv = '../artistSongsLyricsData/azlyrics_lyrics_k.csv'
azLyricsLCsv = '../artistSongsLyricsData/azlyrics_lyrics_l.csv'
azLyricsMCsv = '../artistSongsLyricsData/azlyrics_lyrics_m.csv'
azLyricsNCsv = '../artistSongsLyricsData/azlyrics_lyrics_n.csv'
azLyricsOCsv = '../artistSongsLyricsData/azlyrics_lyrics_o.csv'
azLyricsPCsv = '../artistSongsLyricsData/azlyrics_lyrics_p.csv'
azLyricsQCsv = '../artistSongsLyricsData/azlyrics_lyrics_q.csv'
azLyricsRCsv = '../artistSongsLyricsData/azlyrics_lyrics_r.csv'
azLyricsSCsv = '../artistSongsLyricsData/azlyrics_lyrics_s.csv'
azLyricsTCsv = '../artistSongsLyricsData/azlyrics_lyrics_t.csv'
azLyricsUCsv = '../artistSongsLyricsData/azlyrics_lyrics_u.csv'
azLyricsVCsv = '../artistSongsLyricsData/azlyrics_lyrics_v.csv'
azLyricsWCsv = '../artistSongsLyricsData/azlyrics_lyrics_w.csv'
azLyricsXCsv = '../artistSongsLyricsData/azlyrics_lyrics_x.csv'
azLyricsYCsv = '../artistSongsLyricsData/azlyrics_lyrics_y.csv'
azLyricsZCsv = '../artistSongsLyricsData/azlyrics_lyrics_z.csv'

songsToMeaningCsv = './songsToMeaning.csv'

def getData():

    # concat all the csv files 
    # remove the columns we don't want 
    # do a join on the artist name and song name 
    # return an array of array that contains the artist, song name, song meaning, labels 
    # divide into training and testing 70/30 
    # return training_input, training_labels, test_input, test_labels 
    print('made it')
    df_numbers = pd.read_csv(azLyricsNumbersCsv,error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_a = pd.read_csv(azLyricsACsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_b = pd.read_csv(azLyricsBCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_c = pd.read_csv(azLyricsCCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_d = pd.read_csv(azLyricsDCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_e = pd.read_csv(azLyricsECsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_f = pd.read_csv(azLyricsFCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_g = pd.read_csv(azLyricsGCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_h = pd.read_csv(azLyricsHCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_i = pd.read_csv(azLyricsICsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_j = pd.read_csv(azLyricsJCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_k = pd.read_csv(azLyricsKCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_l = pd.read_csv(azLyricsLCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_m = pd.read_csv(azLyricsMCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_n = pd.read_csv(azLyricsNCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_o = pd.read_csv(azLyricsOCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_p = pd.read_csv(azLyricsPCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_q = pd.read_csv(azLyricsQCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_r = pd.read_csv(azLyricsRCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_s = pd.read_csv(azLyricsSCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_t = pd.read_csv(azLyricsTCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_u = pd.read_csv(azLyricsUCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_v = pd.read_csv(azLyricsVCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_w = pd.read_csv(azLyricsWCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_x = pd.read_csv(azLyricsXCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_y = pd.read_csv(azLyricsYCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_z = pd.read_csv(azLyricsZCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])

    df_songsToLyrics = pd.concat([df_numbers, df_a,df_b,df_c,df_d,df_e,df_f,df_g,df_h,df_i,df_j,df_k,df_l,df_m,df_n,df_o,df_p,df_q,df_r,df_s,df_t,df_u,df_v,df_w,df_x,df_y, df_z])
    df_songsToLyrics = df_songsToLyrics.reset_index(drop=True)
    
    
    df_songsToLyrics = df_songsToLyrics.drop(["ARTIST_URL", "SONG_URL"], axis=1)
    #df_songsToLyrics.to_csv("test_file.csv", sep=",")
    df_songsToMeaning = pd.read_csv(songsToMeaningCsv)
    mergeCsvFiles = df_songsToLyrics.merge(df_songsToMeaning, left_on=["SONG_NAME", "ARTIST_NAME"], right_on=["name", "artist"])
    # numpy array of n songs where each song contains artist name, song name, lyrics and the meaning of the song 
    
    dataNp = mergeCsvFiles[['ARTIST_NAME', 'SONG_NAME', 'LYRICS', 'meaning']].to_numpy()
    
    
    ength = len(dataNp)
    shuffle_idx = np.arange(length)
    np.random.shuffle(shuffle_idx)
    shuffled_inputs = dataNp[shuffle_idx]
    seventyPercent = math.floor(length * .7)
    train = shuffled_inputs[0:seventyPercent]
    test = shuffled_inputs[seventyPercent:]
    train_inputs = [] 
    train_labels = []
    test_inputs = []
    test_labels = []
    
    for elem in train:
        train_inputs.append(elem[0:3])
        train_labels.append(elem[3])
    for elem in test:
        test_inputs.append(elem[0:3])
        test_labels.append(elem[3])

    #print('made it to prints')
    #print(train_inputs)
    #print(train_labels)
    #print(test_inputs)
    #print(test_labels)
    return np.array(train_inputs), np.array(train_labels), np.array(test_inputs), np.array(test_labels)

def getRawData():

    df_numbers = pd.read_csv(azLyricsNumbersCsv,error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_a = pd.read_csv(azLyricsACsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_b = pd.read_csv(azLyricsBCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_c = pd.read_csv(azLyricsCCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_d = pd.read_csv(azLyricsDCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_e = pd.read_csv(azLyricsECsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_f = pd.read_csv(azLyricsFCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_g = pd.read_csv(azLyricsGCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_h = pd.read_csv(azLyricsHCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_i = pd.read_csv(azLyricsICsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_j = pd.read_csv(azLyricsJCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_k = pd.read_csv(azLyricsKCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_l = pd.read_csv(azLyricsLCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_m = pd.read_csv(azLyricsMCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_n = pd.read_csv(azLyricsNCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_o = pd.read_csv(azLyricsOCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_p = pd.read_csv(azLyricsPCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_q = pd.read_csv(azLyricsQCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_r = pd.read_csv(azLyricsRCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_s = pd.read_csv(azLyricsSCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_t = pd.read_csv(azLyricsTCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_u = pd.read_csv(azLyricsUCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_v = pd.read_csv(azLyricsVCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_w = pd.read_csv(azLyricsWCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_x = pd.read_csv(azLyricsXCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_y = pd.read_csv(azLyricsYCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])
    df_z = pd.read_csv(azLyricsZCsv, error_bad_lines=False,names=["ARTIST_NAME","ARTIST_URL","SONG_NAME","SONG_URL","LYRICS"], 
            keep_default_na=False,
            na_values=[''])

    df_songsToLyrics = pd.concat([df_numbers, df_a,df_b,df_c,df_d,df_e,df_f,df_g,df_h,df_i,df_j,df_k,df_l,df_m,df_n,df_o,df_p,df_q,df_r,df_s,df_t,df_u,df_v,df_w,df_x,df_y, df_z])
    df_songsToLyrics = df_songsToLyrics.reset_index(drop=True)
    
    
    df_songsToLyrics = df_songsToLyrics.drop(["ARTIST_URL", "SONG_URL"], axis=1)
    #df_songsToLyrics.to_csv("test_file.csv", sep=",")
    df_songsToMeaning = pd.read_csv(songsToMeaningCsv)
    mergeCsvFiles = df_songsToLyrics.merge(df_songsToMeaning, left_on=["SONG_NAME", "ARTIST_NAME"], right_on=["name", "artist"])
    # numpy array of n songs where each song contains artist name, song name, lyrics and the meaning of the song 
    
    dataNp = mergeCsvFiles[['ARTIST_NAME', 'SONG_NAME', 'LYRICS', 'meaning']].to_numpy()
    return dataNp
    
    
    
    # TODO : 
    # convert data into train, test 70/30
    # then get train_input, train_labels
    # then get test_input, test_labels
    
    # return the 4 results 
    
    
    
    #mergeCsvFiles.to_csv("songsToMeaningAndLyrics.csv", sep = ',')
    #sameArtistMerge = mergeCsvFiles[(mergeCsvFiles['ARTIST_NAME']==mergeCsvFiles['name'])]
    #mergeCsvFiles = pd.merge(df_songsToLyrics, df_songsToMeaning, how='inner', left_on = ["ARTIST_NAME", "SONG_NAME"], right_on = "name", "artist"])
    #mergeCsvFiles.to_csv("songsToMeaningAndLyrics.csv", sep = ',')

getData()