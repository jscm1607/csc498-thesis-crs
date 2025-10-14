# SECTION 1: BASELINES

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

# BASELINE RECOMMENDERS
# Random classifier
def random_recommender(k=10):
    """Recommend k random songs from all training songs."""
    return random.sample(all_songs, k)

# Majority class classifier
def popularity_recommender(k=10):
    """Recommend top-k globally most frequent songs."""
    return list(song_counts.head(k).index)

# EVALUATION
def evaluate_baseline(method_fn, name, n_samples=500, needs_input=False):
    """Compute how often the recommender picks a song in the test playlist."""
    sample = test.sample(n=min(n_samples, len(test)), random_state=16)
    correct = 0

    for _, row in sample.iterrows():
        if needs_input:
            recs = method_fn(row['playlistname'])
        else:
            recs = method_fn()
        true_songs = set(row['song'])
        hits = len(set(recs) & true_songs)
        if hits > 0:
            correct += 1

    accuracy = correct / len(sample)
    print(f"{name:<25} Accuracy: {accuracy:.10f}")
    return accuracy

# RUN
if __name__ == "__main__":
    print("\nBaseline Recommenders Evaluation\n")
    acc_random = evaluate_baseline(random_recommender, "Random Recommender")
    acc_pop = evaluate_baseline(popularity_recommender, "Popularity Recommender")