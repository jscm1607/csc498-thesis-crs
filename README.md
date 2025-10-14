Parameter-Efficient Music Conversational Recommender System
------------

Author: Jose Santiago Campa Morales ([jscm1607@arizona.edu](mailto:jscm1607@arizona.edu))  
Fall 2025 - Spring 2026

## Notes
### Section 1
I downloaded and formatted a playlist dataset and gathered basic stats.<br>
I used the [Spotify Playlists](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)
dataset from Kaggle. Since the dataset is too large for GitHub, please download it in
your local machine under a `data/` folder.

* Number of playlists: 156888 <br>
* Songs per playlist:<br>
count    153832.000000<br>
mean         43.213532<br>
std          65.381311<br>
min           1.000000<br>
25%          11.000000<br>
50%          20.000000<br>
75%          45.000000<br>
max         500.000000

* Playlists per song:<br>
count    2.755287e+06<br>
mean     4.630299e+00<br>
std      2.191614e+01<br>
min      1.000000e+00<br>
25%      1.000000e+00<br>
50%      1.000000e+00<br>
75%      3.000000e+00<br>
max      2.606000e+03

<br>
I also implemented two simple baseline recommender to analyze the dataset.<br>
- Random Recommender: 0% accuracy
- Popularity Recommender: 0.2% accuracy

## References
* [Understanding Baseline Models in Machine Learning](https://medium.com/@preethi_prakash/understanding-baseline-models-in-machine-learning-3ed94f03d645)