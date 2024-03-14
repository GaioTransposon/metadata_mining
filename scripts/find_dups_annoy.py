#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:59:09 2024

@author: dgaio
"""

import os
import pickle
import numpy as np
import time
from annoy import AnnoyIndex
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# --- Loading data ---
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/temp/combined_data.pkl"

# Load embeddings and sample IDs from the file
with open(embeddings_file_path, 'rb') as file:
    data = pickle.load(file)
    
# Convert data to list and slice
sample_ids = list(data.keys())[:3800000]  # Adjust the slice size as needed
all_embeddings = np.array(list(data.values()))[:3800000]  # Slicing to match sample IDs


# =============================================================================
# # Dummy set: 
# all_embeddings = np.random.rand(4000000, 600)
# =============================================================================


# --- PCA for Dimensionality Reduction ---
start_time = time.time()
n_components = 500
pca = PCA(n_components=n_components)
all_embeddings = pca.fit_transform(all_embeddings)
end_time = time.time()
print(f"PCA fitting and transformation took {end_time - start_time:.2f} seconds.")



# --- Building Annoy Index for Nearest Neighbors Search ---
start_time = time.time()
f = len(all_embeddings[0])
n_trees = 10 # more trees give higher precision when querying
index = AnnoyIndex(f, 'angular')    # 'angular', 'euclidean', 'manhattan', 'hamming', or 'dot'
for i, vec in enumerate(all_embeddings):
    index.add_item(i, vec)
index.build(n_trees)
end_time = time.time()
print(f"Building Annoy index took {end_time - start_time:.2f} seconds.")



# --- Nearest Neighbors Search ---
start_time = time.time()
n_neighbors = 11 # Number of nearest neighbors to find (including the query embedding itself)
identical_threshold = 0.1 # Threshold for considering embeddings as nearly identical

def query_batch(start_idx, end_idx):
    local_pairs = []
    for i in range(start_idx, end_idx):
        nearest_neighbors = index.get_nns_by_item(i, n_neighbors, include_distances=True)
        for j, distance in zip(*nearest_neighbors):
            if i != j and distance < identical_threshold:
                local_pairs.append((i, j, distance))
    return local_pairs

batch_size = 1000
num_workers = 4
nearly_identical_pairs = []

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for start_idx in range(0, len(all_embeddings), batch_size):
        end_idx = min(start_idx + batch_size, len(all_embeddings))
        futures.append(executor.submit(query_batch, start_idx, end_idx))
    for future in as_completed(futures):
        nearly_identical_pairs.extend(future.result())

end_time = time.time()
print(f"Nearest neighbors search took {end_time - start_time:.2f} seconds.")



# --- Clustering ---
start_time = time.time()

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
        
n = len(all_embeddings)
parent = [i for i in range(n)]
rank = [0] * n # to keep the tree flat

for idx1, idx2, _ in nearly_identical_pairs:
    union(parent, rank, idx1, idx2)

# Find the representative of each set and group by them, using sample IDs
clusters = {}
for i in range(n):
    rep = find(parent, i)
    sample_id = sample_ids[i]
    if rep in clusters:
        clusters[rep].append(sample_id)
    else:
        clusters[rep] = [sample_id]

# convert clusters from dict to list & filter out singleton clusters
cluster_list = [cluster for cluster in clusters.values() if len(cluster) > 1]

end_time = time.time()
print(f"Clustering took {end_time - start_time:.2f} seconds.")



# --- Summarizing Results ---
print("How many clusters have been created:", len(cluster_list))
total_embeddings = len(all_embeddings)
total_embeddings_in_clusters = sum(len(cluster) for cluster in cluster_list)
total_clusters = len(cluster_list)
samples_saved = total_embeddings - total_embeddings_in_clusters + total_clusters
print("Number of saved samples:", samples_saved)


# --- Inspecting Results ---

# Set a seed for reproducibility
random.seed(21)


def select_clusters_and_calculate_distances(cluster_list, all_embeddings, sample_ids, num_clusters=100, similarity_threshold=0.99):
    # Randomly select clusters, ensuring reproducibility with a fixed seed
    sampled_clusters = random.sample(cluster_list, min(num_clusters, len(cluster_list)))
    
    # Prepare to store clusters and their maximum 'distance' (1 - min similarity)
    cluster_distances = []
    
    for cluster in sampled_clusters:
        # Extract embeddings for the current cluster
        cluster_embeddings = [all_embeddings[sample_ids.index(id)] for id in cluster]
        
        # Calculate cosine similarity matrix for the cluster
        sim_matrix = cosine_similarity(cluster_embeddings)
        
        # Since cosine similarity ranges from -1 (opposite) to 1 (identical), we convert it to 'distance'
        # Distance here is defined as 1 - similarity. We ignore the diagonal (self-similarity) by setting it to 1
        np.fill_diagonal(sim_matrix, 1)
        
        # Find the minimum similarity in the cluster (excluding self-similarity)
        min_similarity = np.min(sim_matrix)
        
        # Convert the minimum similarity to 'distance' and store it along with the cluster
        cluster_distance = 1 - min_similarity
        cluster_distances.append((cluster, cluster_distance))
    
    # Sort clusters by their maximum 'distance' (i.e., where at least one pair has the least similarity)
    cluster_distances.sort(key=lambda x: x[1], reverse=True)
    
    # Return clusters and their 'distances', sorted by the 'distance'
    return [item[0] for item in cluster_distances], [item[1] for item in cluster_distances]



def fetch_metadata_from_sample(sample, path_to_dirs):
    folder_name = f"dir_{sample[-3:]}"  # Assumes directory naming based on sample ID's last 3 chars
    folder_path = os.path.join(path_to_dirs, folder_name)
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path) as f:
        return f.read()

def highlight_differences(text1, text2):
    diffs = []
    for line in set(text1.splitlines()) ^ set(text2.splitlines()):
        diffs.append(f"- {line}") if line in text1 else diffs.append(f"+ {line}")
    return "\n".join(diffs) if diffs else "No significant differences."



def process_clusters(path_to_dirs, cluster_list, all_embeddings, sample_ids, num_clusters):
    
    # Select and calculate distances based on cosine similarity
    sorted_clusters, _ = select_clusters_and_calculate_distances(
        cluster_list, all_embeddings, sample_ids, num_clusters=num_clusters, similarity_threshold=0.99
    )
    
    largest_variance_cluster = sorted_clusters[0]
    cluster_embeddings = [all_embeddings[sample_ids.index(id)] for id in largest_variance_cluster]
    
    # Calculate cosine similarity matrix
    sim_matrix = cosine_similarity(cluster_embeddings)
    
    # Convert similarity to 'distance'
    distance_matrix = 1 - sim_matrix
    
    # Fill diagonal with 0s to ignore self-distance
    np.fill_diagonal(distance_matrix, 0)

    # Dimensionality reduction for 2D visualization using the distance matrix
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=41)
    cluster_2d = mds.fit_transform(distance_matrix)

    # Prepare hover text
    hover_texts = []
    for sample_id in largest_variance_cluster:
        metadata = fetch_metadata_from_sample(sample_id, path_to_dirs)
        brief_metadata_lines = metadata.split('\n')[:10]  # Adjust as needed
        brief_metadata_with_breaks = '<br>'.join(brief_metadata_lines) + '...'
        hover_text = f"Sample ID: {sample_id}<br>{brief_metadata_with_breaks}"
        hover_texts.append(hover_text)

    # Create Plotly figure
    fig = go.Figure(data=go.Scatter(x=cluster_2d[:, 0], y=cluster_2d[:, 1], mode='markers',
                                    marker=dict(size=5),
                                    hoverinfo='text',
                                    text=hover_texts))
    
    # Customize hover label appearance
    fig.update_traces(hoverlabel=dict(namelength=-1,
                                      align='left',
                                      bgcolor='white',
                                      font_size=12,
                                      font_family='Arial, sans-serif'))

    # Show figure
    fig.show()
    fig.write_html('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/largest_distance_cluster_0.1thresh_all_embeddings.html')

    # Show metadata differences in console
    # Find indices of the maximum 'distance' in the distance matrix
    max_dist_indices = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)
    sample_id1, sample_id2 = largest_variance_cluster[max_dist_indices[0]], largest_variance_cluster[max_dist_indices[1]]

    metadata1 = fetch_metadata_from_sample(sample_id1, path_to_dirs)
    metadata2 = fetch_metadata_from_sample(sample_id2, path_to_dirs)
    differences = highlight_differences(metadata1, metadata2)

    print("-" * 50)
    print(f"Metadata for Sample {sample_id1}:\n{metadata1}\n")
    print("-" * 50)
    print(f"Metadata for Sample {sample_id2}:\n{metadata2}\n")
    print("-" * 50)
    print(f"Samples: {sample_id1} and {sample_id2}")
    print("Highlighted Differences:\n")
    print(differences)
    print("-" * 50)




path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"

# Assuming the variables `path_to_dirs`, `cluster_list`, `all_embeddings`, and `sample_ids` are already defined
process_clusters(path_to_dirs, cluster_list, all_embeddings, sample_ids, 100)








