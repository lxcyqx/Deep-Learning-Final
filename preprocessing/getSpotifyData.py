import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

cid = '4a76a5f9d5394199a0830ba20134f062'
secret = '82a8f39ecffd4bc0ae1fb33fe5692131'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

songUniqueIds = set() 

def getTrackIds(playlistTracks):
    ids = []
    for item in playlistTracks['items']:
        #print(item)
        track = item['track']
        #print(track)
        # check song has not already been included for another label or the same label 
        if track['id'] not in songUniqueIds:
            ids.append(track['id'])
    return ids

def getTrackFeatures(id):
    meta = sp.track(id)
    name = meta['name']
    artist = meta['album']['artists'][0]['name']
    track = [name, artist]
    return track

def allTracksWithFeatures(playlistIds):
    tracksWithFeature = [] 
    for id in playlistIds:
        trackFeatures = getTrackFeatures(id)
        tracksWithFeature.append(trackFeatures)
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

lovePlaylistSongNameAndArtist = allTracksWithFeatures(allLovePlaylistTrackIds)

# create dataset
print("made it ")
dfLovePlaylist = pd.DataFrame(lovePlaylistSongNameAndArtist , columns = ['name', 'artist'])
dfLovePlaylist.to_csv("lovePlaylist.csv", sep = ',')
print("created")
        
# track_dataframe = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'popularity' : popularity})
# print(track_dataframe.shape)
# track_dataframe.head()