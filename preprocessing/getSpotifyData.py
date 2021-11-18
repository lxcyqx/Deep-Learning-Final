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

# love playlist #1 https://open.spotify.com/playlist/2p6SAq9WHFaodqizQ0TTdO

playlists = sp.user_playlists('spotify')

for i in range(0,10000,50):
    track_results = sp.search(q='year:2018', type='track', limit=50,offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        popularity.append(t['popularity'])
        
        
track_dataframe = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'popularity' : popularity})
print(track_dataframe.shape)
track_dataframe.head()