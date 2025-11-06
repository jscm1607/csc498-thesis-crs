"""
Spotify Playlist Recommendation Baselines
------------------------------------------
This script implements several baseline recommenders for playlist continuation:
1. Random and Popularity baselines
2. TF-IDF and BM25 baselines
3. Word2Vec embedding-based baseline
Evaluation metrics: Precision@K, Recall@K, F1@K, Hit Rate, MRR, MAP, NDCG
"""

import pandas as pd
import numpy as np
import random
import ast
import inspect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi
from collections import Counter
from sklearn.metrics import ndcg_score

# =====================================================
# === SECTION 0: DATA LOADING AND PREPROCESSING =====
# =====================================================

def safe_parse_list(s):
    """Safely parse string or list-like cells into Python lists."""
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                return ast.literal_eval(s)
            except Exception:
                return [s]
        else:
            return [item.strip() for item in s.split(',')]
    elif isinstance(s, list):
        return s
    else:
        return []

def load_and_prepare_data(path="data/spotify_dataset.csv"):
    """Load and preprocess Spotify dataset."""
    df = pd.read_csv(path, quotechar='"', sep=',', on_bad_lines='skip', engine='python')

    # Clean headers
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Combine trackname and artistname
    df['song'] = df['trackname'] + ' - ' + df['artistname']

    # Drop NaNs and parse songs
    df = df.dropna(subset=['playlistname', 'song'])
    df['song'] = df['song'].apply(safe_parse_list)

    # Filter out large playlists
    df = df[df['song'].apply(len) <= 500]

    # Split train/test
    train = df.sample(frac=0.80, random_state=16)
    test = df.drop(train.index)

    return df, train, test

# =====================================================
# === DATA SUMMARY ===================================
# =====================================================

def summarize_data(df):
    """Print dataset summary statistics."""
    playlists = df.groupby('playlistname')['song'].apply(list).reset_index()
    num_playlists = len(playlists)
    playlists = playlists[playlists['song'].apply(len) <= 500]

    # Songs per playlist
    songs_per_playlist = playlists['song'].apply(len)
    songs_per_playlist_stats = songs_per_playlist.describe(percentiles=[.25, .5, .75])

    # Playlists per song
    songs_flat = df['trackname'] + ' - ' + df['artistname']
    playlists_per_song = songs_flat.value_counts()
    playlists_per_song_stats = playlists_per_song.describe(percentiles=[.25, .5, .75])

    print(f"Number of playlists: {num_playlists}")
    print("\nSongs per playlist:")
    print(songs_per_playlist_stats)
    print("\nPlaylists per song:")
    print(playlists_per_song_stats)

# =====================================================
# === METRIC DEFINITIONS =============================
# =====================================================

def precision_at_k(predicted, ground_truth, k):
    relevant = set(ground_truth)
    return len(set(predicted[:k]) & relevant) / k

def recall_at_k(predicted, ground_truth, k):
    relevant = set(ground_truth)
    return len(set(predicted[:k]) & relevant) / len(relevant) if len(relevant) > 0 else 0

def f1_at_k(predicted, ground_truth, k):
    p = precision_at_k(predicted, ground_truth, k)
    r = recall_at_k(predicted, ground_truth, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0

def hit_rate(predicted, ground_truth, k):
    return 1.0 if len(set(predicted[:k]) & set(ground_truth)) > 0 else 0.0

def mrr(predicted, ground_truth):
    for rank, item in enumerate(predicted, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0

def average_precision(predicted, ground_truth, k):
    relevant = set(ground_truth)
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:k], start=1):
        if p in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if len(relevant) > 0 else 0

def ndcg_at_k(predicted, ground_truth, k):
    y_true = [[1 if item in ground_truth else 0 for item in predicted[:k]]]
    y_score = [[1.0 / (i + 1) for i in range(k)]]
    return ndcg_score(y_true, y_score)


def evaluate_recommender(method_fn, test_df, name, k=10):
    metrics = {
        "Precision@K": [],
        "Recall@K": [],
        "F1@K": [],
        "HitRate": [],
        "MRR": [],
        "MAP": [],
        "NDCG": [],
    }

    for _, row in test_df.iterrows():
        # Determine if the recommender expects an argument
        sig = inspect.signature(method_fn)
        if len(sig.parameters) == 0:
            # Random or Popularity recommender: no input needed
            recs = method_fn()
        else:
            # TF-IDF, BM25, Word2Vec: pass the playlist
            recs = method_fn(row['text'] if 'text' in row else row['song'], k=k)

        ground_truth = row['song']

        metrics["Precision@K"].append(precision_at_k(recs, ground_truth, k))
        metrics["Recall@K"].append(recall_at_k(recs, ground_truth, k))
        metrics["F1@K"].append(f1_at_k(recs, ground_truth, k))
        metrics["HitRate"].append(hit_rate(recs, ground_truth, k))
        metrics["MRR"].append(mrr(recs, ground_truth))
        metrics["MAP"].append(average_precision(recs, ground_truth, k))
        metrics["NDCG"].append(ndcg_at_k(recs, ground_truth, k))

    avg_metrics = {m: np.mean(v) for m, v in metrics.items()}
    print(f"\n{name:<25}")
    for m, v in avg_metrics.items():
        print(f"{m:<15}: {v:.4f}")

    return avg_metrics


# =====================================================
# === SECTION 1: SIMPLE BASELINES ====================
# =====================================================

def run_simple_baselines(train, test_sample):
    all_songs = train.explode('song')['song'].tolist()
    song_counts = pd.Series(all_songs).value_counts()

    def random_recommender(playlist=None, k=10):
        return random.sample(all_songs, k)

    def popularity_recommender(playlist=None, k=10):
        return list(song_counts.head(k).index)

    print("\n--- SIMPLE BASELINES ---")
    eval_random = evaluate_recommender(random_recommender, test_sample, "Random Recommender", k=10)
    eval_pop = evaluate_recommender(popularity_recommender, test_sample, "Popularity Recommender", k=10)

    return eval_random, eval_pop

# =====================================================
# === SECTION 2: TF-IDF & BM25 BASELINES =============
# =====================================================

def run_text_baselines(train, test_sample):
    def playlist_to_text(row):
        return " ".join(row['song'])

    train_playlists = (
        train.groupby('playlistname')['song']
        .apply(lambda x: [s for sublist in x for s in sublist])
        .reset_index()
    )
    train_playlists['text'] = train_playlists.apply(playlist_to_text, axis=1)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500)
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_playlists['text'])

    def tfidf_recommender(test_text, k=10):
        test_vec = tfidf_vectorizer.transform([test_text])
        scores = cosine_similarity(test_vec, tfidf_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        recs = []
        for idx in top_idx:
            recs.extend(train_playlists.loc[idx, 'song'])
        return list(dict.fromkeys(recs))[:k]

    # BM25
    tokenized_corpus = [text.split() for text in train_playlists['text']]
    bm25 = BM25Okapi(tokenized_corpus)

    def bm25_recommender(test_text, k=10):
        tokens = test_text.split()
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        recs = []
        for idx in top_idx:
            recs.extend(train_playlists.loc[idx, 'song'])
        return list(dict.fromkeys(recs))[:k]

    print("\n--- TEXT-BASED BASELINES ---")
    eval_tfidf = evaluate_recommender(tfidf_recommender, test_sample, "TF-IDF Recommender", k=10)
    eval_bm25 = evaluate_recommender(bm25_recommender, test_sample, "BM25 Recommender", k=10)

    return eval_tfidf, eval_bm25

# =====================================================
# === SECTION 3: WORD2VEC BASELINE ===================
# =====================================================

def run_word2vec_baseline(train, test_sample):
    # Prepare playlists
    train_playlists = (
        train.groupby('playlistname')['song']
        .apply(lambda x: [s for sublist in x for s in sublist])
        .reset_index()
    )

    # Train Word2Vec
    sentences = train_playlists['song'].tolist()
    w2v_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4, sg=1)

    # Playlist vector (average of song embeddings)
    def get_playlist_vector(playlist):
        vectors = [w2v_model.wv[s] for s in playlist if s in w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(w2v_model.vector_size)
        return np.mean(vectors, axis=0)

    # Precompute train playlist vectors
    train_vectors = np.array([get_playlist_vector(p) for p in train_playlists['song']])

    # Recommender function (no recursion!)
    # Wrap the recommender so it can accept either a playlist list or the
    # precomputed `text` string that `evaluate_recommender` may pass. If a
    # string is received, try to recover the original song list by matching
    # against `test_sample['text']`; fall back to splitting if not found.
    def w2v_recommender(test_playlist_input, k=10):
        # Normalize input to a list of song strings
        if isinstance(test_playlist_input, str):
            # Attempt to recover the original playlist from the provided
            # test_sample (matching on the 'text' field). This handles the
            # case where evaluate_recommender passes row['text'] instead of
            # row['song'].
            matched = test_sample[test_sample['text'] == test_playlist_input]
            if not matched.empty:
                test_playlist = matched.iloc[0]['song']
            else:
                # Last-resort: split on whitespace (may be lossy) so we don't
                # crash — this is better than producing zero vectors.
                test_playlist = test_playlist_input.split()
        else:
            test_playlist = test_playlist_input

        test_vec = get_playlist_vector(test_playlist).reshape(1, -1)
        scores = cosine_similarity(test_vec, train_vectors)[0]

        # Take top N playlists
        top_n = 30
        top_idx = np.argsort(scores)[::-1][:top_n]

        # Aggregate songs weighted by playlist similarity
        song_scores = Counter()
        for idx in top_idx:
            sim = scores[idx]
            for song in train_playlists.loc[idx, 'song']:
                song_scores[song] += sim

        # Return top-k songs
        return [song for song, _ in song_scores.most_common(k)]

    # Evaluate
    print("\n--- WORD2VEC BASELINE ---")
    eval_w2v = evaluate_recommender(w2v_recommender, test_sample, "Word2Vec Recommender", k=10)
    
    return eval_w2v

# =====================================================
# === MAIN EXECUTION ==================================
# =====================================================

if __name__ == "__main__":
    df, train, test = load_and_prepare_data()
    summarize_data(df)

    # --- Prepare a single test sample ---
    test_playlists = (
        test.groupby('playlistname')['song']
        .apply(lambda x: [s for sublist in x for s in sublist])
        .reset_index()
    )
    test_playlists['text'] = test_playlists['song'].apply(lambda x: " ".join(x))
    test_sample = test_playlists.sample(n=min(100, len(test_playlists)), random_state=16)

    # --- Run all baselines using K = 10 ---
    print("\n--- RUNNING BASELINES (K=10) ---")
    eval_random, eval_pop = run_simple_baselines(train, test_sample)
    eval_tfidf, eval_bm25 = run_text_baselines(train, test_sample)
    eval_w2v = run_word2vec_baseline(train, test_sample)

    # --- Plotting and results saving ---
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- Collect metrics for plotting ---
    methods = ['Random', 'Popularity', 'TF-IDF', 'BM25', 'Word2Vec']
    evals = [eval_random, eval_pop, eval_tfidf, eval_bm25, eval_w2v]

    metrics_df = pd.DataFrame(evals, index=methods).reset_index().rename(columns={'index': 'method'})

    # --- Bar charts for each metric at K=10 ---
    for metric in ['Precision@K', 'Recall@K', 'F1@K', 'NDCG', 'HitRate', 'MRR', 'MAP']:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='method', y=metric, data=metrics_df)
        plt.title(f'{metric} at K=10 by Method')
        plt.ylabel(metric)
        plt.xlabel('Method')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f'figures/{metric.lower()}_at_10_bar.png', dpi=200)
        plt.close()

    # --- Histogram of playlist lengths (test set) ---
    playlist_lengths = test_playlists['song'].apply(len)
    plt.figure(figsize=(8, 6))
    sns.histplot(playlist_lengths, bins=30, kde=True)
    plt.title('Histogram of Playlist Lengths (Test Set)')
    plt.xlabel('Number of Songs')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('figures/playlist_length_histogram.png', dpi=200)
    plt.close()

    # --- Example recommendation results (first 5 playlists) ---
    example_rows = []
    for idx, row in test_playlists.head(5).iterrows():
        playlist = row['song']
        gt = playlist
        recs = {}
        recs['Random'] = run_simple_baselines(train, test_playlists.iloc[[idx]])[0]['Precision@K']
        recs['Popularity'] = run_simple_baselines(train, test_playlists.iloc[[idx]])[1]['Precision@K']
        recs['TF-IDF'] = run_text_baselines(train, test_playlists.iloc[[idx]])[0]['Precision@K']
        recs['BM25'] = run_text_baselines(train, test_playlists.iloc[[idx]])[1]['Precision@K']
        recs['Word2Vec'] = run_word2vec_baseline(train, test_playlists.iloc[[idx]])['Precision@K']
        example_rows.append({
            'playlistname': row['playlistname'],
            'ground_truth': ", ".join(gt),
            **{f'precision_{m}': recs[m] for m in recs}
        })
    pd.DataFrame(example_rows).to_csv('results/example_recommendations.csv', index=False)

    # --- Metric correlation heatmap ---
    corr_df = metrics_df[['Precision@K', 'Recall@K', 'F1@K', 'NDCG', 'HitRate', 'MRR', 'MAP']]
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Metric Correlations at K=10')
    plt.tight_layout()
    plt.savefig('figures/metric_correlation_heatmap.png', dpi=200)
    plt.close()