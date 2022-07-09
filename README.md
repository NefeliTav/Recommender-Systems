# Recommendation Systems

```
python recommendation.py
```

We tried all prediction algorithms offered by the Surprise library, (Basic, k-NN, Matrix Factorization) and then we compared different configurations for KNNBaseline and SVD.

# Pagerank

```
python pagerank.py
```

We created a graph of Pokemon and computed the Topic-Specific Pagerank for a different topic each time, using the Networkx library. Also, we proved that this procedure builds teams not simply by aggregating teams generated from individual nodes.

# Communities

```
python communities.py
```

Using the same Pokemon graph, we found the local communities around them, using Personalized Pagerank.

# Embeddings

```
python embeddings.py
```

Get word embeddings from text , train and test two models and then classify claims as REFUTES or SUPPORTS.
