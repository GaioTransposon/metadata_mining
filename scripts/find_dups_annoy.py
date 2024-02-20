#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:59:09 2024

@author: dgaio
"""


import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import difflib
import time
import networkx as nx
from sklearn.preprocessing import normalize
from annoy import AnnoyIndex
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import numpy as np
import pickle
from sklearn.decomposition import PCA




# Path to the embeddings file
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/combined_data.pkl"

# Load embeddings and sample IDs from the file
with open(embeddings_file_path, 'rb') as file:
    data = pickle.load(file)
    
    
    
# Convert data to list and slice
sample_ids = list(data.keys())[:250000]  # Adjust the slice size as needed
all_embeddings = np.array(list(data.values()))[:250000]  # Slicing to match sample IDs





# Number of components to keep
n_components = 256  # Example value, adjust based on your needs

# Initialize PCA
pca = PCA(n_components=n_components)

# Fit PCA on your embeddings and transform the embeddings
all_embeddings = pca.fit_transform(all_embeddings)

# pca_embeddings now contains the reduced dimensionality embeddings






f = len(all_embeddings[1])  # Dimensionality of the embeddings
n_trees = 10  # More trees give higher precision when querying
index = AnnoyIndex(f, 'angular')  # Use 'angular', 'euclidean', 'manhattan', 'hamming', or 'dot'

for i, vec in enumerate(all_embeddings):
    index.add_item(i, vec)

index.build(n_trees)



n_neighbors = 11  # Number of nearest neighbors to find (including the query embedding itself)
identical_threshold = 0.1  # Threshold for considering embeddings as nearly identical, adjust based on your needs


from concurrent.futures import ThreadPoolExecutor, as_completed

def query_batch(start_idx, end_idx):
    local_pairs = []
    for i in range(start_idx, end_idx):
        nearest_neighbors = index.get_nns_by_item(i, n_neighbors, include_distances=True)
        for j, distance in zip(*nearest_neighbors):
            if i != j and distance < identical_threshold:
                local_pairs.append((i, j, distance))
    return local_pairs

# Define batch size and number of workers
batch_size = 1000  # Adjust based on your system's capabilities
num_workers = 4  # Number of threads
nearly_identical_pairs = []

# Create ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for start_idx in range(0, len(all_embeddings), batch_size):
        end_idx = min(start_idx + batch_size, len(all_embeddings))
        futures.append(executor.submit(query_batch, start_idx, end_idx))

    # Collect results
    for future in as_completed(futures):
        nearly_identical_pairs.extend(future.result())

len(nearly_identical_pairs)





def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

# Number of embeddings
n = len(all_embeddings)

# Parent array where parent[i] is the parent (or representative) of i
parent = [i for i in range(n)]

# Rank array used to keep the tree flat
rank = [0] * n

for idx1, idx2, _ in nearly_identical_pairs:
    union(parent, rank, idx1, idx2)

# Find the representative of each set and group by them, using sample IDs
clusters = {}
for i in range(n):
    rep = find(parent, i)
    sample_id = sample_ids[i]  # Convert index to sample ID
    if rep in clusters:
        clusters[rep].append(sample_id)
    else:
        clusters[rep] = [sample_id]

# Convert clusters from dict to list for easier usage
cluster_list = [cluster for cluster in clusters.values()]

print(len(cluster_list))

total_embeddings = len(all_embeddings)  # Total number of embeddings
total_embeddings_in_clusters = sum(len(cluster) for cluster in cluster_list)  # Total number of embeddings in clusters
total_clusters = len(cluster_list)  # Total number of clusters

# Calculate the number of samples "saved" by clustering
samples_saved = total_embeddings - total_embeddings_in_clusters + total_clusters

print(samples_saved)







