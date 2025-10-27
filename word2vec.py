# SECTION 3: WORD2VEC BASELINE

import pandas as pd
import numpy as np
import ast

# Part 1: Read and clean csv
df = pd.read_csv(
    "data/spotify_dataset.csv",
    quotechar='"',
    sep=',',
    on_bad_lines='skip',
    engine='python'
)

#print(df.head())
#print(df.columns)

# clean headers
df.columns = df.columns.str.replace('"', '').str.strip()

# create song column combining tracks and artists
df['song'] = df['trackname'] + ' - ' + df['artistname']

# create playlist list
playlists = (
    df.groupby('playlistname')['song']
      .apply(list)
      .reset_index()
)

# Part 2: Create baselines
# Get valid data
df = df.dropna(subset=['playlistname', 'song'])

# String representation of list to actual list
def safe_parse_list(s):
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                # ast.literal_eval
                return ast.literal_eval(s)
            except Exception:
                return [s]
        else:
            # comma-separated string
            return [item.strip() for item in s.split(',')]
    # already in list -- just return
    elif isinstance(s, list):
        return s
    else:
        return []

# Create valid song values
df['song'] = df['song'].apply(safe_parse_list)

# Filter outliers
df = df[df['song'].apply(len) <= 500]

# Train/test split
train = df.sample(frac=0.90, random_state=16)
test = df.drop(train.index)

# Count all instances of songs
all_songs = train.explode('song')['song'].tolist()
song_counts = pd.Series(all_songs).value_counts()

# Convert list of songs into a single string per playlist
def playlist_to_text(row):
    return " ".join(row['song'])

train_playlists = train.groupby('playlistname')['song'].apply(lambda x: [s for sublist in x for s in sublist]).reset_index()
train_playlists['text'] = train_playlists.apply(playlist_to_text, axis=1)

test_playlists = test.groupby('playlistname')['song'].apply(lambda x: [s for sublist in x for s in sublist]).reset_index()
test_playlists['text'] = test_playlists.apply(playlist_to_text, axis=1)
test_playlists_sample = test_playlists.sample(n=100, random_state=16)

from gensim.models import Word2Vec

# Each playlist is a "sentence", each song is a "word"
sentences = train_playlists['song'].tolist()

# Train Word2Vec
w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1
)

def get_playlist_vector(playlist, model):
    vectors = [model.wv[song] for song in playlist if song in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

train_vectors = np.array([get_playlist_vector(p, w2v_model) for p in train_playlists['song']])


from sklearn.metrics.pairwise import cosine_similarity

def w2v_recommender(test_playlist, train_playlists, train_vectors, model, k=10):
    test_vec = get_playlist_vector(test_playlist, model).reshape(1, -1)
    scores = cosine_similarity(test_vec, train_vectors)[0]
    top_idx = np.argsort(scores)[::-1][:k]
    recommendations = []
    for idx in top_idx:
        recommendations.extend(train_playlists.loc[idx, 'song'])
    return list(dict.fromkeys(recommendations))[:k]

correct = 0
for _, row in test_playlists_sample.iterrows():
    recs = w2v_recommender(row['song'], train_playlists, train_vectors, w2v_model, k=10)
    true_songs = set(row['song'])
    if len(set(recs) & true_songs) > 0:
        correct += 1
accuracy = correct / len(test_playlists_sample)
print(f"Word2Vec Recommender Accuracy: {accuracy:.4f}")