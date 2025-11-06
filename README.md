Parameter-Efficient Music Conversational Recommender System
------------

Author: Jose Santiago Campa Morales ([jscm1607@arizona.edu](mailto:jscm1607@arizona.edu))  
Fall 2025 - Spring 2026

## Overview
This project explores baseline music recommender systems using using the 
[Spotify Playlists](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists) dataset
from Kaggle.<br>
Because the dataset is too large for GitHub, please download it locally and place it under
a `data/` directory:<br>
`data/spotify_dataset.csv`

## Notes
### Dataset Statistics and Basic Baselines
I downloaded and formatted the playlist dataset and gathered basic statistics.
For consistency, I limited playlist sizes to 500 songs.<br>

#### Dataset Summary
Number of playlists: 156706

Songs per playlist:
count    153660
mean         43.201764
std          65.370672
min           1
25%          11
50%          20
75%          45
max         500

Playlists per song:
count    2755285
mean     4.630287
std      21.91595
min      1
25%      1
50%      1
75%      3
max      2606

##### Baseline Recommenders
--- SIMPLE BASELINES ---

Random Recommender
Precision@K    : 0.0000
Recall@K       : 0.0000
F1@K           : 0.0000
HitRate        : 0.0000
MRR            : 0.0000
MAP            : 0.0000
NDCG           : 0.0000

Popularity Recommender
Precision@K    : 0.0000
Recall@K       : 0.0000
F1@K           : 0.0000
HitRate        : 0.0000
MRR            : 0.0000
MAP            : 0.0000
NDCG           : 0.0000

--- TEXT-BASED BASELINES ---

TF-IDF Recommender
Precision@K    : 0.0900
Recall@K       : 0.1634
F1@K           : 0.0970
HitRate        : 0.3900
MRR            : 0.1603
MAP            : 0.0759
NDCG           : 0.2263

BM25 Recommender
Precision@K    : 0.0970
Recall@K       : 0.2014
F1@K           : 0.1055
HitRate        : 0.4300
MRR            : 0.1847
MAP            : 0.0856
NDCG           : 0.2387

--- WORD2VEC BASELINE ---

Word2Vec Recommender
Precision@K    : 0.0470
Recall@K       : 0.0625
F1@K           : 0.0443
HitRate        : 0.2800
MRR            : 0.1038
MAP            : 0.0283
NDCG           : 0.1524

## References
* [Understanding Baseline Models in Machine Learning](https://medium.com/@preethi_prakash/understanding-baseline-models-in-machine-learning-3ed94f03d645)
* [TF-IDF vs BM25: Understanding and Implementing Text Ranking Algorithms in Python](https://medium.com/@macikgozm/tf-idf-vs-bm25-understanding-and-implementing-text-ranking-algorithms-in-python-f56111f5086b)
* [What is cosine similarity?](https://www.ibm.com/think/topics/cosine-similarity)
* [A Dummy’s Guide to Word2Vec](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)
* [10 metrics to evaluate recommender and ranking systems](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)