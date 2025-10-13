import pandas as pd
import numpy as np


df = pd.read_csv(
    "data/spotify_dataset.csv",
    quotechar='"',    # recognize quotes properly
    sep=',',          # ensure comma delimiter
    on_bad_lines='skip',  # skip bad rows instead of crashing (optional)
    engine='python'   # more forgiving parser than C engine
)

#print(df.head())
#print(df.columns)

df.columns = df.columns.str.replace('"', '').str.strip()

df['song'] = df['trackname'] + ' - ' + df['artistname']
playlists = (
    df.groupby('playlistname')['song']
      .apply(list)
      .reset_index()
)

#print(playlists)

num_playlists = len(playlists)

# Remove big playlists -- from max 1.331416e+06 to max 500
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

# PART 2: BASELINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# ============================================================
# 1. LOAD DATA
# ============================================================
# Remove missing or malformed entries
df = df.dropna(subset=['playlistname', 'song'])

import ast

# Convert string representation of list to actual list safely
def safe_parse_list(s):
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                # Use ast.literal_eval instead of eval
                return ast.literal_eval(s)
            except Exception:
                # fallback: treat as single string
                return [s]
        else:
            # comma-separated string fallback
            return [item.strip() for item in s.split(',')]
    elif isinstance(s, list):
        return s
    else:
        return []

df['song'] = df['song'].apply(safe_parse_list)

# Filter extreme outliers
df = df[df['song'].apply(len) <= 500]

# Train/test split (e.g., 95/5%)
train = df.sample(frac=0.95, random_state=42)
test = df.drop(train.index)

# Flatten song occurrences for global popularity
all_songs = [song for songs in train['song'] for song in songs]
song_counts = pd.Series(all_songs).value_counts()

# ============================================================
# 2. BASELINE RECOMMENDERS # https://medium.com/@preethi_prakash/understanding-baseline-models-in-machine-learning-3ed94f03d645
# ============================================================

# Random classifier
def random_recommender(k=10):
    """Recommend k random songs from all training songs."""
    return random.sample(all_songs, k)

# Majority class classifier
def popularity_recommender(k=10):
    """Recommend top-k globally most frequent songs."""
    return list(song_counts.head(k).index)

# ============================================================
# 3. EVALUATION
# ============================================================
def evaluate_baseline(method_fn, name, k=10, n_samples=200, needs_input=False):
    """Compute how often the recommender picks a song in the test playlist."""
    sample = test.sample(n=min(n_samples, len(test)), random_state=42)
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
    print(f"{name:<25} Accuracy: {accuracy:.5f}")
    return accuracy

# ============================================================
# 4. RUN EXPERIMENTS
# ============================================================
if __name__ == "__main__":
    print("\n🎧 Baseline Recommenders Evaluation\n" + "-"*40)
    acc_random = evaluate_baseline(random_recommender, "Random Recommender")
    acc_pop = evaluate_baseline(popularity_recommender, "Popularity Recommender")

    print("\n✅ Done. Results Summary:")
    print(f"Random: {acc_random:.5f}")
    print(f"Popularity: {acc_pop:.5f}")