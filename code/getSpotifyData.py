import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Make HTTP requests
import requests
# Scrape data from an HTML document
from bs4 import BeautifulSoup
# I/O
import os
# Search and manipulate strings
import re
import numpy as np
import math


#GENIUS API STUFF
GENIUS_API_TOKEN='YcO4NV1shV8O0pA4kw9eCJflks0JpecSoMpU2v8sc2fitjDfKfHpGOClBJSAzstM'

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
    replaceName = name.replace(" ", "-")
    newStr = initialStr + replaceArtist + "-" + replaceName + "-lyrics"
    
    return scrape_song_lyrics(newStr)
    

cid = '4a76a5f9d5394199a0830ba20134f062'
secret = '82a8f39ecffd4bc0ae1fb33fe5692131'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

songUniqueIds = {}

def getTrackIds(playlistTracks):
    ids = []
    for item in playlistTracks['items']:
        track = item['track']

        # check song has not already been included for another label or the same label 
        #if track['id'] not in songUniqueIds:
        ids.append(track['id'])
    return ids

def getTrackFeatures(id, label):
    meta = sp.track(id)
    name = meta['name']
    artist = meta['album']['artists'][0]['name']
    track = [name.lower(), artist.lower(),label]
    return track

def allTracksWithFeatures(playlistIds,label):
    tracksWithFeature = [] 
    for id in playlistIds:
        trackFeatures = getTrackFeatures(id, label)
        name, artist = trackFeatures[0], trackFeatures[1]
        # I need to form the str "artist name separated by -, -, song name separated by - "
        initialStr = "https://genius.com/"
        # consider removing punctation from artist name and song name 
        replaceArtist = artist.replace(" ","-")
        replaceName = name.replace(" ", "-")
        newStr = initialStr + replaceArtist + "-" + replaceName + "-lyrics"
        checkStr = name + artist 

        if checkStr not in songUniqueIds:
            lyrics = scrape_song_lyrics(newStr)
            if lyrics is not None:
                tracksWithFeature.append([trackFeatures[1], trackFeatures[0], lyrics, trackFeatures[2]])
                songUniqueIds[checkStr] = True
    return tracksWithFeature

# I need to search for the playlist I want 
# use sp.search (to find specific playlist)

# love playlist #1 Love songs for him/her! https://open.spotify.com/playlist/2p6SAq9WHFaodqizQ0TTdO
# love playlist #2 Timeless Love Songs https://open.spotify.com/playlist/37i9dQZF1DX7rOY2tZUw1k
# love playlist #3 love https://open.spotify.com/playlist/7tj5Dblki89mIdHVzIm6DO
# love playlist #4 Country Kind Of Love https://open.spotify.com/playlist/37i9dQZF1DX8WMG8VPSOJC
# love playlist #5 Love songs https://open.spotify.com/playlist/7BqzGVMQ21rr63avfYYnn0
# love playlist #6 Rock Love Songs https://open.spotify.com/playlist/37i9dQZF1DX7Z7kYpKKGTc
# love playlist #7 Love Pop https://open.spotify.com/playlist/37i9dQZF1DX50QitC6Oqtn
# love playlist #8 Love Ballads https://open.spotify.com/playlist/37i9dQZF1DWYMvTygsLWlG
# love playlist #9 Love Letter https://open.spotify.com/playlist/37i9dQZF1DX38lOuCWlLV1
# love playlist #10 80s Love Song https://open.spotify.com/playlist/37i9dQZF1DXc3KygMa1OE7
# love playlist #11 70s Love Song https://open.spotify.com/playlist/37i9dQZF1DWY373eEGlSj4
# love playlist #12 90s Love Song https://open.spotify.com/playlist/37i9dQZF1DWXqpDKK4ed9O
# love playlist #13 love songs but in an r&b playlist https://open.spotify.com/playlist/4s8rDkXIwUzfY03HlA6VfR
# love playlist #14 Love song 80s 90s https://open.spotify.com/playlist/7a1xivxbS2IUmDs6sE3l8T
# love playlist #15 Love Songs 2021 https://open.spotify.com/playlist/4pqdXECBpzbBQNBQ7IrPi2

lovePlaylistOneTracks = sp.playlist_tracks("https://open.spotify.com/playlist/2p6SAq9WHFaodqizQ0TTdO", additional_types=('track', ))
lovePlaylistTwoTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX7rOY2tZUw1k", additional_types=('track', ))
lovePlaylistThreeTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7tj5Dblki89mIdHVzIm6DO", additional_types=('track', ))
lovePlaylistFourTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX8WMG8VPSOJC", additional_types=('track', ))
lovePlaylistFiveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7BqzGVMQ21rr63avfYYnn0", additional_types=('track', ))
lovePlaylistSixTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX7Z7kYpKKGTc", additional_types=('track', ))
lovePlaylistSevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX50QitC6Oqtn", additional_types=('track', ))
lovePlaylistEightTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DWYMvTygsLWlG", additional_types=('track', ))
lovePlaylistNineTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX38lOuCWlLV1", additional_types=('track', ))
lovePlaylistTenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DXc3KygMa1OE7", additional_types=('track', ))
lovePlaylistElevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DWY373eEGlSj4", additional_types=('track', ))
lovePlaylistTwelveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DWXqpDKK4ed9O", additional_types=('track', ))
lovePlaylistThirteenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/4s8rDkXIwUzfY03HlA6VfR", additional_types=('track', ))
lovePlaylistFourteenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7a1xivxbS2IUmDs6sE3l8T", additional_types=('track', ))
lovePlaylistFifteenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/4pqdXECBpzbBQNBQ7IrPi2", additional_types=('track', ))


lovePlaylistOneTrackIds = getTrackIds(lovePlaylistOneTracks)
lovePlaylistTwoTrackIds = getTrackIds(lovePlaylistTwoTracks)
lovePlaylistThreeTrackIds = getTrackIds(lovePlaylistThreeTracks)
lovePlaylistFourTrackIds = getTrackIds(lovePlaylistFourTracks)
lovePlaylistFiveTrackIds = getTrackIds(lovePlaylistFiveTracks)
lovePlaylistSixTrackIds = getTrackIds(lovePlaylistSixTracks)
lovePlaylistSevenTrackIds = getTrackIds(lovePlaylistSevenTracks)
lovePlaylistEightTrackIds = getTrackIds(lovePlaylistEightTracks)
lovePlaylistNineTrackIds = getTrackIds(lovePlaylistNineTracks)
lovePlaylistTenTrackIds = getTrackIds(lovePlaylistTenTracks)
lovePlaylistElevenTrackIds = getTrackIds(lovePlaylistElevenTracks)
lovePlaylistTwelveTrackIds = getTrackIds(lovePlaylistTwelveTracks)
lovePlaylistThirteenTrackIds = getTrackIds(lovePlaylistThirteenTracks)
lovePlaylistFourteenTrackIds = getTrackIds(lovePlaylistFourteenTracks)
lovePlaylistFifteenTrackIds = getTrackIds(lovePlaylistFifteenTracks)

allLovePlaylistTrackIds = lovePlaylistOneTrackIds + lovePlaylistTwoTrackIds + lovePlaylistThreeTrackIds + lovePlaylistFourTrackIds + lovePlaylistFiveTrackIds + lovePlaylistSixTrackIds  + lovePlaylistSevenTrackIds + lovePlaylistEightTrackIds + lovePlaylistNineTrackIds + lovePlaylistTenTrackIds + lovePlaylistElevenTrackIds + lovePlaylistTwelveTrackIds + lovePlaylistThirteenTrackIds + lovePlaylistFourteenTrackIds + lovePlaylistFifteenTrackIds  
lovePlaylistSongNameAndArtist = allTracksWithFeatures(allLovePlaylistTrackIds, 'love')

# breakup playlist #1 breakp songs to scream in the car https://open.spotify.com/playlist/5WXBDWaVjp0Wbcnj6oW9Bk
# breakup playlist #2 break up songs for sad bitches https://open.spotify.com/playlist/2L0KREU9VNMVXmuasVIpik
# breakup playlist #3 Fuck You Break Up Songs https://open.spotify.com/playlist/7q7pTBSWD35yGaf44V5SY2
# breakup playlist #4 Breakup SONGS FOR WHEN MY.. https://open.spotify.com/playlist/2bklWGt6k5NWx1SCjfnJGb
# breakup playlist #5 Breakup country songs  https://open.spotify.com/playlist/7L3dhElX5zyKGQbUCBkIX9
# breakup playlist #6 Angry Break up songs https://open.spotify.com/playlist/1UFbWIAY8e3HLZBmSPjVmW
# breakup playlist #7 breakup songs that hit hard https://open.spotify.com/playlist/56HIOy1LVUyqvKSy9QcdfS
# breakup playlist #8 breakup songs for girls https://open.spotify.com/playlist/7yaeboe0JxH8fIwQhGxnUl
# breakup playlist #9 glow up after breakup https://open.spotify.com/playlist/1DW8iudbAvG9fAtnA5oFpw
# breakup playlist #10 break up songs https://open.spotify.com/playlist/7lP9dVzRbxGVWw21G0UrV8
# breakup playlist #11 bad bitch break up playlist https://open.spotify.com/playlist/7p6oSuF8WZajIXBMNHT2Td
# breakup playlist #12 break up playlist r&b https://open.spotify.com/playlist/3F1J6kT566Fbnxw4IFO0Pv


breakupPlaylistOneTracks = sp.playlist_tracks("https://open.spotify.com/playlist/5WXBDWaVjp0Wbcnj6oW9Bk", additional_types=('track', ))
breakupPlaylistTwoTracks = sp.playlist_tracks("https://open.spotify.com/playlist/2L0KREU9VNMVXmuasVIpik", additional_types=('track', ))
breakupPlaylistThreeTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7q7pTBSWD35yGaf44V5SY2", additional_types=('track', ))
breakupPlaylistFourTracks = sp.playlist_tracks("https://open.spotify.com/playlist/2bklWGt6k5NWx1SCjfnJGb", additional_types=('track', ))
breakupPlaylistFiveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7L3dhElX5zyKGQbUCBkIX9", additional_types=('track', ))
breakupPlaylistSixTracks = sp.playlist_tracks("https://open.spotify.com/playlist/1UFbWIAY8e3HLZBmSPjVmW", additional_types=('track', ))
breakupPlaylistSevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/56HIOy1LVUyqvKSy9QcdfS", additional_types=('track', ))
breakupPlaylistEightTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7yaeboe0JxH8fIwQhGxnUl", additional_types=('track', ))
breakupPlaylistNineTracks = sp.playlist_tracks("https://open.spotify.com/playlist/1DW8iudbAvG9fAtnA5oFpw", additional_types=('track', ))
breakupPlaylistTenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7lP9dVzRbxGVWw21G0UrV8", additional_types=('track', ))
breakupPlaylistElevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7p6oSuF8WZajIXBMNHT2Td", additional_types=('track', ))
breakupPlaylistTwelveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/3F1J6kT566Fbnxw4IFO0Pv", additional_types=('track', ))


breakupPlaylistOneTrackIds = getTrackIds(breakupPlaylistOneTracks)
breakupPlaylistTwoTrackIds = getTrackIds(breakupPlaylistTwoTracks)
breakupPlaylistThreeTrackIds = getTrackIds(breakupPlaylistThreeTracks)
breakupPlaylistFourTrackIds = getTrackIds(breakupPlaylistFourTracks)
breakupPlaylistFiveTrackIds = getTrackIds(breakupPlaylistFiveTracks)
breakupPlaylistSixTrackIds = getTrackIds(breakupPlaylistSixTracks)
breakupPlaylistSevenTrackIds = getTrackIds(breakupPlaylistSevenTracks)
breakupPlaylistEightTrackIds = getTrackIds(breakupPlaylistEightTracks)
breakupPlaylistNineTrackIds = getTrackIds(breakupPlaylistNineTracks)
breakupPlaylistTenTrackIds = getTrackIds(breakupPlaylistTenTracks)
breakupPlaylistElevenTrackIds = getTrackIds(breakupPlaylistElevenTracks)
breakupPlaylistTwelveTrackIds = getTrackIds(breakupPlaylistTwelveTracks)

allBreakupPlaylistTrackIds = breakupPlaylistOneTrackIds + breakupPlaylistTwoTrackIds + breakupPlaylistThreeTrackIds + breakupPlaylistFourTrackIds + breakupPlaylistFiveTrackIds + breakupPlaylistSixTrackIds  + breakupPlaylistSevenTrackIds + breakupPlaylistEightTrackIds + breakupPlaylistNineTrackIds + breakupPlaylistTenTrackIds + breakupPlaylistElevenTrackIds + breakupPlaylistTwelveTrackIds 
breakupPlaylistSongNameAndArtist = allTracksWithFeatures(allBreakupPlaylistTrackIds, 'breakup')

# party playlist #1 Party Songs https://open.spotify.com/playlist/5xS3Gi0fA3Uo6RScucyct6
# party playlist #2 80s party https://open.spotify.com/playlist/37i9dQZF1DX6xnkAwJX7tn
# party playlist #3 Halloween Party https://open.spotify.com/playlist/37i9dQZF1DX8S9gwdi7dev
# party playlist #4 Rock Party https://open.spotify.com/playlist/37i9dQZF1DX8FwnYE6PRvL
# party playlist #5 Party Country  https://open.spotify.com/playlist/5omTfsfJJs8qwUAwFxbSGm
# party playlist #6 Party Songs 2000-2021 https://open.spotify.com/playlist/57mNC119tTcPulrOwxIjhn
# party playlist #7 90s Party https://open.spotify.com/playlist/37i9dQZF1DXdo6A3mWpdWx
# party playlist #8 Party Songs every one should know https://open.spotify.com/playlist/59yjm9k3adI8vXl3m4OkjF
# party playlist #9 party Bangers Only https://open.spotify.com/playlist/6m0VPA0AqL26IKhrSMizvv
# party playlist #10 Party Hits 2000s https://open.spotify.com/playlist/37i9dQZF1DX7e8TjkFNKWH
# party playlist #11 Party Playlist https://open.spotify.com/playlist/7E8Qv0JN7bvBOE5fzpHqTt
# party playlist #12 College Party Music 2022 https://open.spotify.com/playlist/6rzpktRxXPgzu0yqGWxDC4

partyPlaylistOneTracks = sp.playlist_tracks("https://open.spotify.com/playlist/5xS3Gi0fA3Uo6RScucyct6", additional_types=('track', ))
partyPlaylistTwoTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX6xnkAwJX7tn", additional_types=('track', ))
partyPlaylistThreeTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX8S9gwdi7dev", additional_types=('track', ))
partyPlaylistFourTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX8FwnYE6PRvL", additional_types=('track', ))
partyPlaylistFiveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/5omTfsfJJs8qwUAwFxbSGm", additional_types=('track', ))
partyPlaylistSixTracks = sp.playlist_tracks("https://open.spotify.com/playlist/57mNC119tTcPulrOwxIjhn", additional_types=('track', ))
partyPlaylistSevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DXdo6A3mWpdWx", additional_types=('track', ))
partyPlaylistEightTracks = sp.playlist_tracks("https://open.spotify.com/playlist/59yjm9k3adI8vXl3m4OkjF", additional_types=('track', ))
partyPlaylistNineTracks = sp.playlist_tracks("https://open.spotify.com/playlist/6m0VPA0AqL26IKhrSMizvv", additional_types=('track', ))
partyPlaylistTenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX7e8TjkFNKWH", additional_types=('track', ))
partyPlaylistElevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7E8Qv0JN7bvBOE5fzpHqTt", additional_types=('track', ))
partyPlaylistTwelveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/6rzpktRxXPgzu0yqGWxDC4", additional_types=('track', ))


partyPlaylistOneTrackIds = getTrackIds(partyPlaylistOneTracks)
partyPlaylistTwoTrackIds = getTrackIds(partyPlaylistTwoTracks)
partyPlaylistThreeTrackIds = getTrackIds(partyPlaylistThreeTracks)
partyPlaylistFourTrackIds = getTrackIds(partyPlaylistFourTracks)
partyPlaylistFiveTrackIds = getTrackIds(partyPlaylistFiveTracks)
partyPlaylistSixTrackIds = getTrackIds(partyPlaylistSixTracks)
partyPlaylistSevenTrackIds = getTrackIds(partyPlaylistSevenTracks)
partyPlaylistEightTrackIds = getTrackIds(partyPlaylistEightTracks)
partyPlaylistNineTrackIds = getTrackIds(partyPlaylistNineTracks)
partyPlaylistTenTrackIds = getTrackIds(partyPlaylistTenTracks)
partyPlaylistElevenTrackIds = getTrackIds(partyPlaylistElevenTracks)
partyPlaylistTwelveTrackIds = getTrackIds(partyPlaylistTwelveTracks)

allPartyPlaylistTrackIds = partyPlaylistOneTrackIds + partyPlaylistTwoTrackIds + partyPlaylistThreeTrackIds + partyPlaylistFourTrackIds + partyPlaylistFiveTrackIds + partyPlaylistSixTrackIds  + partyPlaylistSevenTrackIds + partyPlaylistEightTrackIds + partyPlaylistNineTrackIds + partyPlaylistTenTrackIds + partyPlaylistElevenTrackIds + partyPlaylistTwelveTrackIds 
partyPlaylistSongNameAndArtist = allTracksWithFeatures(allPartyPlaylistTrackIds, 'party')

# sex playlist #1 sex https://open.spotify.com/playlist/5i0m2whLHnXlytw3JDARca
# sex playlist #2 Dirty Sex Songs https://open.spotify.com/playlist/7gkbtCFu5DFXfNjB00RVDb
# sex playlist #3 Sexy Country Songs https://open.spotify.com/playlist/1E1V1lxrFM56gBDsSP0oR8
# sex playlist #4 Sex Playlist 2021 https://open.spotify.com/playlist/2clZBI7A4jFQHOHo3KCCHG
# sex playlist #5 sex  https://open.spotify.com/playlist/0JT8Oqj3U3GGmHDZ5kKqCS
# sex playlist #6 sexy timezz https://open.spotify.com/playlist/6wwf7rIkaUTVESgzcInJ3t
# sex playlist #7 Sexy Time 2021 https://open.spotify.com/playlist/5aT4806I753UfrawGOOK9z
# sex playlist #8 sexXx playlist https://open.spotify.com/playlist/3Di5mi72pDBWk1bFFOhoc0
# sex playlist #9 Sex Playlist https://open.spotify.com/playlist/3PLdzvyOaVrvOnuGT08Cb6
# sex playlist #10 Sensual Songs https://open.spotify.com/playlist/7ebQdxl0t7c8kr2nwFxOaJ
# sex playlist #11 Sex Songs 2021 https://open.spotify.com/playlist/5INRJHYwsLHOcgsKr0fR01
# sex playlist #12 Music for sex https://open.spotify.com/playlist/0cvittMyMe4cu3lKMXb96m

sexPlaylistOneTracks = sp.playlist_tracks("https://open.spotify.com/playlist/5i0m2whLHnXlytw3JDARca", additional_types=('track', ))
sexPlaylistTwoTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7gkbtCFu5DFXfNjB00RVDb", additional_types=('track', ))
sexPlaylistThreeTracks = sp.playlist_tracks("https://open.spotify.com/playlist/1E1V1lxrFM56gBDsSP0oR8", additional_types=('track', ))
sexPlaylistFourTracks = sp.playlist_tracks("https://open.spotify.com/playlist/2clZBI7A4jFQHOHo3KCCHG", additional_types=('track', ))
sexPlaylistFiveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/0JT8Oqj3U3GGmHDZ5kKqCS", additional_types=('track', ))
sexPlaylistSixTracks = sp.playlist_tracks("https://open.spotify.com/playlist/6wwf7rIkaUTVESgzcInJ3t", additional_types=('track', ))
sexPlaylistSevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/5aT4806I753UfrawGOOK9z", additional_types=('track', ))
sexPlaylistEightTracks = sp.playlist_tracks("https://open.spotify.com/playlist/3Di5mi72pDBWk1bFFOhoc0", additional_types=('track', ))
sexPlaylistNineTracks = sp.playlist_tracks("https://open.spotify.com/playlist/3PLdzvyOaVrvOnuGT08Cb6", additional_types=('track', ))
sexPlaylistTenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7ebQdxl0t7c8kr2nwFxOaJ", additional_types=('track', ))
sexPlaylistElevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/5INRJHYwsLHOcgsKr0fR01", additional_types=('track', ))
sexPlaylistTwelveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/0cvittMyMe4cu3lKMXb96m", additional_types=('track', ))


sexPlaylistOneTrackIds = getTrackIds(sexPlaylistOneTracks)
sexPlaylistTwoTrackIds = getTrackIds(sexPlaylistTwoTracks)
sexPlaylistThreeTrackIds = getTrackIds(sexPlaylistThreeTracks)
sexPlaylistFourTrackIds = getTrackIds(sexPlaylistFourTracks)
sexPlaylistFiveTrackIds = getTrackIds(sexPlaylistFiveTracks)
sexPlaylistSixTrackIds = getTrackIds(sexPlaylistSixTracks)
sexPlaylistSevenTrackIds = getTrackIds(sexPlaylistSevenTracks)
sexPlaylistEightTrackIds = getTrackIds(sexPlaylistEightTracks)
sexPlaylistNineTrackIds = getTrackIds(sexPlaylistNineTracks)
sexPlaylistTenTrackIds = getTrackIds(sexPlaylistTenTracks)
sexPlaylistElevenTrackIds = getTrackIds(sexPlaylistElevenTracks)
sexPlaylistTwelveTrackIds = getTrackIds(sexPlaylistTwelveTracks)

allSexPlaylistTrackIds = sexPlaylistOneTrackIds + sexPlaylistTwoTrackIds + sexPlaylistThreeTrackIds + sexPlaylistFourTrackIds + sexPlaylistFiveTrackIds + sexPlaylistSixTrackIds  + sexPlaylistSevenTrackIds + sexPlaylistEightTrackIds + sexPlaylistNineTrackIds + sexPlaylistTenTrackIds + sexPlaylistElevenTrackIds + sexPlaylistTwelveTrackIds 
sexPlaylistSongNameAndArtist = allTracksWithFeatures(allSexPlaylistTrackIds, 'sex')

# religion playlist #1 Religious Songs https://open.spotify.com/playlist/1Lfv5hpiBqtxHIlaeUo8TS
# religion playlist #2 paster kid religious trauma https://open.spotify.com/playlist/7hhyLxjaqqYGGdJWftG2oS
# religion playlist #3 religious trauma fueled god https://open.spotify.com/playlist/4Yy1zdlFWr0lgX5PAYIQGN
# religion playlist #4 Top Christian Worship Songs https://open.spotify.com/playlist/30wdzfOKmW7JmLPiQI0BGC
# religion playlist #5 Christian Music 2021  https://open.spotify.com/playlist/14YGydApBed973QAGn4geK
# religion playlist #6 Roman Catholic Hymns https://open.spotify.com/playlist/4FJzDjKjsx9kUREzEZYIX9
# religion playlist #7 Catholic Praise & Worship https://open.spotify.com/playlist/4NAcDUNDiMArjdZHaC6DVl
# religion playlist #8 religious trauma https://open.spotify.com/playlist/7izMgSSUJm5a5PcwSIEIXP

religionPlaylistOneTracks = sp.playlist_tracks("https://open.spotify.com/playlist/1Lfv5hpiBqtxHIlaeUo8TS", additional_types=('track', ))
religionPlaylistTwoTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7hhyLxjaqqYGGdJWftG2oS", additional_types=('track', ))
religionPlaylistThreeTracks = sp.playlist_tracks("https://open.spotify.com/playlist/4Yy1zdlFWr0lgX5PAYIQGN", additional_types=('track', ))
religionPlaylistFourTracks = sp.playlist_tracks("https://open.spotify.com/playlist/30wdzfOKmW7JmLPiQI0BGC", additional_types=('track', ))
religionPlaylistFiveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/14YGydApBed973QAGn4geK", additional_types=('track', ))
religionPlaylistSixTracks = sp.playlist_tracks("https://open.spotify.com/playlist/4FJzDjKjsx9kUREzEZYIX9", additional_types=('track', ))
religionPlaylistSevenTracks = sp.playlist_tracks("https://open.spotify.com/playlist/4NAcDUNDiMArjdZHaC6DVl", additional_types=('track', ))
religionPlaylistEightTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7izMgSSUJm5a5PcwSIEIXP", additional_types=('track', ))

religionPlaylistOneTrackIds = getTrackIds(religionPlaylistOneTracks)
religionPlaylistTwoTrackIds = getTrackIds(religionPlaylistTwoTracks)
religionPlaylistThreeTrackIds = getTrackIds(religionPlaylistThreeTracks)
religionPlaylistFourTrackIds = getTrackIds(religionPlaylistFourTracks)
religionPlaylistFiveTrackIds = getTrackIds(religionPlaylistFiveTracks)
religionPlaylistSixTrackIds = getTrackIds(religionPlaylistSixTracks)
religionPlaylistSevenTrackIds = getTrackIds(religionPlaylistSevenTracks)
religionPlaylistEightTrackIds = getTrackIds(religionPlaylistEightTracks)

allReligionPlaylistTrackIds = religionPlaylistOneTrackIds + religionPlaylistTwoTrackIds + religionPlaylistThreeTrackIds + religionPlaylistFourTrackIds + religionPlaylistFiveTrackIds + religionPlaylistSixTrackIds  + religionPlaylistSevenTrackIds + religionPlaylistEightTrackIds 
religionPlaylistSongNameAndArtist = allTracksWithFeatures(allReligionPlaylistTrackIds, 'religion')


allSongArtistMeaning = lovePlaylistSongNameAndArtist+ breakupPlaylistSongNameAndArtist + partyPlaylistSongNameAndArtist + sexPlaylistSongNameAndArtist + religionPlaylistSongNameAndArtist

# create dataset - 
dfPlaylist = pd.DataFrame(allSongArtistMeaning, columns = ['ARTIST_NAME', 'SONG_NAME', 'LYRICS', 'meaning'])
dfPlaylist.to_csv("songsToMeaningAndLyricsReal.csv", sep = ',')


def returnData():
    dfPlayShortcut = pd.read_csv("./songsToMeaningAndLyricsReal.csv")
    dataNp = dfPlayShortcut[['ARTIST_NAME', 'SONG_NAME', 'LYRICS', 'meaning']].to_numpy()
    length = len(dataNp)
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
    return dataNp, train_inputs, train_labels, test_inputs,test_labels


