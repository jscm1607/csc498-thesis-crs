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
### Section 1 - Dataset Statistics and Basic Baselines
I downloaded and formatted the playlist dataset and gathered basic statistics.
For consistency, I limited playlist sizes to 500 songs.<br>

#### Dataset Summary
* Number of playlists: 156888 <br>
* Songs per playlist:<br>
count    153832<br>
mean         43.21<br>
std          65.38<br>
min           1<br>
25%          11<br>
50%          20<br>
75%          45<br>
max         500

* Playlists per song:<br>
count    2755287<br>
mean           4.63<br>
std            21.92<br>
min            1<br>
25%            1<br>
50%            1<br>
75%            3<br>
max         2606

##### Baseline Recommenders
- Random Recommender (Select a random song): 0% accuracy<br>
- Popularity Recommender (Select most popular song): 0% accuracy

### Section 2 - More Sophisticated Baselines
Now, I tested two more sophisticated recommenders.<br>

- TF-IDF Recommender (vector-based): 40% accuracy<br>
- BM25 Recommender (probabilistic ranking): 41% accuracy<br>

To use BM25, install the package below:<br>
`pip install rank_bm25`

## References
* [Understanding Baseline Models in Machine Learning](https://medium.com/@preethi_prakash/understanding-baseline-models-in-machine-learning-3ed94f03d645)
* [TF-IDF vs BM25: Understanding and Implementing Text Ranking Algorithms in Python](https://medium.com/@macikgozm/tf-idf-vs-bm25-understanding-and-implementing-text-ranking-algorithms-in-python-f56111f5086b)