# SECTION 2: MORE SOPHISTICATED BASELINES

import pandas as pd
import numpy as np
import random
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

#print(playlists)

# Part 2: Get playlist stats
num_playlists = len(playlists)

# Remove big playlists -- from max 1331416 to max 500
playlists = playlists[playlists['song'].apply(len) <= 500]

# Songs per playlist
songs_per_playlist = playlists['song'].apply(len)
songs_per_playlist_stats = songs_per_playlist.describe(percentiles=[.25, .5, .75])

# Playlists per song
songs_flat = df['trackname'] + ' - ' + df['artistname']
playlists_per_song = songs_flat.value_counts()
playlists_per_song_stats = playlists_per_song.describe(percentiles=[.25, .5, .75])

# Summary
print(f"Number of playlists: {num_playlists}")
print("\nSongs per playlist:")
print(songs_per_playlist_stats)
print("\nPlaylists per song:")
print(playlists_per_song_stats)

# PART 3: Create baselines
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

# Count all instances of songs for popularity rec
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

# IF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2500)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_playlists['text'])

def tfidf_recommender(test_text, k=10):
    test_vec = tfidf_vectorizer.transform([test_text])
    scores = cosine_similarity(test_vec, tfidf_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:k]
    recommendations = []
    for idx in top_idx:
        recommendations.extend(train_playlists.loc[idx, 'song'])
    # Deduplicate
    return list(dict.fromkeys(recommendations))[:k]

# BM25
from rank_bm25 import BM25Okapi

tokenized_corpus = [text.split() for text in train_playlists['text']]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_recommender(test_text, k=10):
    tokens = test_text.split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    recommendations = []
    for idx in top_idx:
        recommendations.extend(train_playlists.loc[idx, 'song'])
    return list(dict.fromkeys(recommendations))[:k]

# EVALUATION
def evaluate_baselines(recommender_fn, name, test_df=test_playlists_sample, k=10):
    correct = 0
    for _, row in test_df.iterrows():
        recs = recommender_fn(row['text'], k=k)
        true_songs = set(row['song'])
        hits = len(set(recs) & true_songs)
        if hits > 0:
            correct += 1
    accuracy = correct / len(test_df)
    print(f"{name:<25} Accuracy: {accuracy:.10f}")
    return accuracy

# RUN
if __name__ == "__main__":
    print("\nBaseline Recommenders Evaluation\n")
    acc_tdidf = evaluate_baselines(tfidf_recommender, "TF-IDF Recommender")
    acc_bm25 = evaluate_baselines(bm25_recommender, "BM25 Recommender")