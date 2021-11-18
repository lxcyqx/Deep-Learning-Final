import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

cid = '4a76a5f9d5394199a0830ba20134f062'
secret = '82a8f39ecffd4bc0ae1fb33fe5692131'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

artist_name = []
track_name = []
popularity = []
track_id = []
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
lovePlaylistOneTrackIds = getTrackIds(lovePlaylistOneTracks)


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




tracks = []
for i in range(len(ids)):
    time.sleep(.5)
    track = getTrackFeatures(ids[i])
    tracks.append(track)

# create dataset
df = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])
df.to_csv("spotify.csv", sep = ',')



# def getTrackIDs(user, playlist_id):
#     ids = []
#     playlist = sp.user_playlist(user, playlist_id)
#     for item in playlist['tracks']['items']:
#         track = item['track']
#         ids.append(track['id'])
#     return ids



# for i in range(0,10000,50):
#     track_results = sp.search(q='year:2018', type='track', limit=50,offset=i)
#     for i, t in enumerate(track_results['tracks']['items']):
#         artist_name.append(t['artists'][0]['name'])
#         track_name.append(t['name'])2
#         track_id.append(t['id'])
#         popularity.append(t['popularity'])
        
        
# track_dataframe = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'popularity' : popularity})
# print(track_dataframe.shape)
# track_dataframe.head()