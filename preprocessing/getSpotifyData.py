import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

cid = '4a76a5f9d5394199a0830ba20134f062'
secret = '82a8f39ecffd4bc0ae1fb33fe5692131'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# I need to search for the playlist I want 
# use sp.search (to find specific playlist)

# love playlist #1 Love songs for him/her! https://open.spotify.com/playlist/2p6SAq9WHFaodqizQ0TTdO
# love playlist #2 Timeless Love Songs https://open.spotify.com/playlist/37i9dQZF1DX7rOY2tZUw1k
# love playlist #3 love https://open.spotify.com/playlist/7tj5Dblki89mIdHVzIm6DO
# love playlist #4 Country Kind Of Love https://open.spotify.com/playlist/37i9dQZF1DX8WMG8VPSOJC
# love playlist #5 Love songs https://open.spotify.com/playlist/7BqzGVMQ21rr63avfYYnn0
# love playlist #6 Rock Love Songs https://open.spotify.com/playlist/37i9dQZF1DX7Z7kYpKKGTc

songUniqueIds = set() 

lovePlaylistOneTracks = sp.playlist_tracks("https://open.spotify.com/playlist/2p6SAq9WHFaodqizQ0TTdO", additional_types=('track', ))
lovePlaylistTwoTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX7rOY2tZUw1k", additional_types=('track', ))
lovePlaylistThreeTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7tj5Dblki89mIdHVzIm6DO", additional_types=('track', ))
lovePlaylistFourTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX8WMG8VPSOJC", additional_types=('track', ))
lovePlaylistFiveTracks = sp.playlist_tracks("https://open.spotify.com/playlist/7BqzGVMQ21rr63avfYYnn0", additional_types=('track', ))
lovePlaylistSixTracks = sp.playlist_tracks("https://open.spotify.com/playlist/37i9dQZF1DX7Z7kYpKKGTc", additional_types=('track', ))

lovePlaylistOneTrackIds = getTrackIds(lovePlaylistOneTracks)
lovePlaylistTwoTrackIds = getTrackIds(lovePlaylistTwoTracks)
lovePlaylistThreeTrackIds = getTrackIds(lovePlaylistThreeTracks)
lovePlaylistFourTrackIds = getTrackIds(lovePlaylistFourTracks)
lovePlaylistFiveTrackIds = getTrackIds(lovePlaylistFiveTracks)
lovePlaylistSixTrackIds = getTrackIds(lovePlaylistSixTracks)
allLovePlaylistTrackIds = lovePlaylistOneTrackIds + lovePlaylistTwoTrackIds + lovePlaylistThreeTrackIds + lovePlaylistFourTrackIds + lovePlaylistFiveTrackIds + lovePlaylistSixTrackIds 
lovePlaylistSongNameAndArtist = allTracksWithFeatures(allLovePlaylistTrackIds)


def getTrackIds(playlistTracks):
    ids = []
    for track in playlistTracks:
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
    

# create dataset
dfLovePlaylist = pd.DataFrame(lovePlaylistSongNameAndArtist , columns = ['name', 'artist'])
dfLovePlaylist.to_csv("lovePlaylist.csv", sep = ',')

        
# track_dataframe = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'popularity' : popularity})
# print(track_dataframe.shape)
# track_dataframe.head()