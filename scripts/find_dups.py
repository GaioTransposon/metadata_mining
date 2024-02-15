#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:40:08 2024

@author: dgaio
"""

import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import matplotlib.pyplot as plt
import difflib
import time
import networkx as nx
from sklearn.preprocessing import normalize



# Path to the embeddings file
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/combined_data.pkl"

# Load embeddings and sample IDs from the file
with open(embeddings_file_path, 'rb') as file:
    data = pickle.load(file)
    
    
    
# Convert data to list and slice
sample_ids = list(data.keys())[:10000]  # Adjust the slice size as needed
all_embeddings = np.array(list(data.values()))[:10000]  # Slicing to match sample IDs








# 1. 

start = time.time()

# Calculate cosine similarity matrix for the subset
similarity_matrix = cosine_similarity(all_embeddings)

# Set a threshold for considering embeddings as 'nearly identical'
similarity_threshold = 0.99

# Find pairs of similar embeddings based on the threshold
similar_pairs = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):  # Compare each pair only once
        if similarity_matrix[i, j] >= similarity_threshold:
            similar_pairs.append((sample_ids[i], sample_ids[j], similarity_matrix[i, j]))

# Create a graph
G = nx.Graph()

# Add edges for pairs with high similarity
for sample1, sample2, score in similar_pairs:
    if score >= 0.99:  # similarity threshold
        G.add_edge(sample1, sample2)

# Find connected components, which represent groups of similar samples
connected_components = list(nx.connected_components(G))

# Create a dictionary to hold groups of similar samples
groups = {}
for i, component in enumerate(connected_components):
    group_name = f'group_{i + 1}'
    groups[group_name] = list(component)

# Print the groups
for group_name, samples in groups.items():
    pass
    #print(f"{group_name}: {samples}")

end = time.time()
print(end-start)





# 2. 

start = time.time()

# Normalize the embeddings
normalized_embeddings = normalize(all_embeddings)

# Optional: Sort embeddings by a density measure, such as the sum of vector components
sorted_indices = np.argsort(np.sum(normalized_embeddings, axis=1))[::-1]
sorted_embeddings = normalized_embeddings[sorted_indices]
sorted_sample_ids = [sample_ids[i] for i in sorted_indices]  # Map sorted indices to sample IDs

clusters = []  # Each cluster will contain sample IDs
representatives = []  # List of representative embeddings for each cluster

for idx, embedding in enumerate(sorted_embeddings):
    found_cluster = False
    for rep_idx, representative in enumerate(representatives):
        similarity = np.dot(embedding, representative)  # Dot product since vectors are normalized
        if similarity >= 0.88:  # Adjusted similarity threshold
            clusters[rep_idx].append(sorted_sample_ids[idx])  # Use sample ID instead of index
            found_cluster = True
            break
    if not found_cluster:
        representatives.append(embedding)
        clusters.append([sorted_sample_ids[idx]])  # Start a new cluster with the sample ID

# Convert the list of clusters into a dictionary with named groups
groups_method2 = {f'group_{i + 1}': cluster for i, cluster in enumerate(clusters)}

end = time.time()
print(end-start)









import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time

def cluster_kmeans(embeddings, n_clusters=100):
    start_time = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=100)
    kmeans.fit(embeddings)
    duration = time.time() - start_time
    return kmeans.labels_, duration


import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import time

def cluster_kmeans_with_threshold(embeddings, n_clusters, threshold_percentile):
    start_time = time.time()

    # Optional: Scale embeddings - Uncomment if scaling is desired
    # scaler = StandardScaler()
    # embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=0, batch_size=100)
    kmeans.fit(embeddings)  # Change to embeddings_scaled if scaling is applied

    # Calculate distances from each point to its cluster center
    distances = cdist(embeddings, kmeans.cluster_centers_, 'euclidean')
    min_distances = np.min(distances, axis=1)

    # Determine the 99th percentile of distances
    threshold_distance = np.percentile(min_distances, threshold_percentile)

    # Identify valid clusters
    valid_clusters = []
    for i in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) < 2:
            # Skip clusters with fewer than 2 embeddings
            continue

        cluster_distances = min_distances[cluster_indices]
        if np.all(cluster_distances < threshold_distance):
            valid_clusters.append(i)

    duration = time.time() - start_time

    # Filter labels to include only valid clusters, set others to -1 for 'noise' or standalone
    valid_labels = np.array([-1 if label not in valid_clusters else label for label in kmeans.labels_])

    return valid_labels, valid_clusters, duration


def cluster_kmeans_with_threshold_and_ids(embeddings, sample_ids, n_clusters, threshold_percentile):
    start_time = time.time()

    # Optional: Scale embeddings - Uncomment if scaling is desired
    # scaler = StandardScaler()
    # embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=0, batch_size=100)
    kmeans.fit(embeddings)  # Change to embeddings_scaled if scaling is applied

    # Calculate distances from each point to its cluster center
    distances = cdist(embeddings, kmeans.cluster_centers_, 'euclidean')
    min_distances = np.min(distances, axis=1)

    # Determine the 99th percentile of distances
    threshold_distance = np.percentile(min_distances, threshold_percentile)

    # Identify valid clusters and their sample IDs
    valid_clusters = {}
    for i in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) < 2:
            # Skip clusters with fewer than 2 embeddings
            continue

        cluster_distances = min_distances[cluster_indices]
        if np.all(cluster_distances < threshold_distance):
            # Only include clusters where all points are within the threshold distance
            valid_clusters[i] = [sample_ids[idx] for idx in cluster_indices]

    duration = time.time() - start_time

    return valid_clusters, duration





# Assuming your embeddings are loaded into `all_embeddings`

labels_kmeans, duration_kmeans = cluster_kmeans(all_embeddings)
print(f"K-means clustering took {duration_kmeans} seconds.")


valid_labels, valid_clusters, duration = cluster_kmeans_with_threshold(all_embeddings, 10, 90)

valid_clusters, duration = cluster_kmeans_with_threshold_and_ids(all_embeddings, sample_ids, 100, 90)










































# =============================================================================
# # compare groups: 
# 
# print(groups)
# print(len(groups))
# 
# print(groups_method2)
# print(len(groups_method2))
# 
# 
# def get_identical_clusters(clusters1, clusters2):
#     # Convert clusters to frozensets for efficient, immutable, and hashable set operations
#     sets1 = [frozenset(cluster) for cluster in clusters1.values()]
#     sets2 = [frozenset(cluster) for cluster in clusters2.values()]
# 
#     # Use a set to avoid counting duplicates
#     matched_clusters = set()
# 
#     for set1 in sets1:
#         for set2 in sets2:
#             if set1 == set2:  # Check if two clusters are exactly the same
#                 matched_clusters.add(tuple(set1))  # Add to the set of matched clusters
# 
#     # The number of matched clusters
#     return matched_clusters
# 
# # Assuming `groups` from Method 1 and `groups_method2` from Method 2 contain the actual clusters
# identical_clusters_count = get_identical_clusters(groups, groups_method2)
# idc=len(identical_clusters_count)
# 
# print(f"Identical Clusters: {identical_clusters_count}, {idc}")
# =============================================================================


# =============================================================================
# # Plotting similarities: 
# 
# # Initialize an empty list to store the similarity scores excluding self-comparisons
# non_self_similarity_scores = []
# 
# # Loop over the similarity matrix and collect non-self similarity scores
# for i in range(len(similarity_matrix)):
#     for j in range(i + 1, len(similarity_matrix)):  # This ensures we only consider each pair once and exclude self-comparisons
#         non_self_similarity_scores.append(similarity_matrix[i, j])
# 
# 
# 
# # Define bin edges to align with desired ticks
# bin_edges = np.linspace(0, 1, 51)  # Creates 50 bins between 0 and 1
# bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
# 
# # Plot the histogram with defined bin edges
# plt.hist(non_self_similarity_scores, bins=bin_edges, color='skyblue', edgecolor='gray')
# 
# # Set ticks at bin centers
# plt.xticks(bin_centers, rotation=90)  # Rotate ticks for better visibility
# 
# # Labeling the plot
# plt.title('Distribution of Non-Self Cosine Similarity Scores')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Frequency')
# plt.grid(True)
# 
# # Show plot
# plt.tight_layout()  # Adjust layout to not cut off labels
# plt.show()
# =============================================================================



# =============================================================================
# # Save these pairs and their similarity scores to a CSV
# with open('similar_samples.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Sample 1 ID', 'Sample 2 ID', 'Similarity Score'])
#     writer.writerows(similar_pairs)
# =============================================================================


# =============================================================================
# import cProfile
# import pstats
# 
# with cProfile.Profile() as pr:
#     similarity_matrix = cosine_similarity(all_embeddings)
# 
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
# =============================================================================


# =============================================================================
# # Showing diffs in metadata:
# 
# 
# # Utility to fetch metadata from folders
# def fetch_metadata_from_sample(sample, path_to_dirs):
#     folder_name = f"dir_{sample[-3:]}"  # Assumes the last 3 characters of the sample ID indicate the directory
#     folder_path = os.path.join(path_to_dirs, folder_name)
#     metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
#     with open(metadata_file_path, 'r') as f:
#         metadata = f.read()
#     return metadata
# 
# 
# 
# # Function to highlight differences more concisely
# def highlight_differences(text1, text2):
#     # Split the texts into lines
#     text1_lines = text1.splitlines()
#     text2_lines = text2.splitlines()
# 
#     # Find lines in text1 that are not in text2, and vice versa
#     unique_to_text1 = [line for line in text1_lines if line not in text2_lines]
#     unique_to_text2 = [line for line in text2_lines if line not in text1_lines]
# 
#     # Use difflib to find close matches and highlight the differences
#     diffs = []
#     for line in unique_to_text1:
#         close_matches = difflib.get_close_matches(line, unique_to_text2, n=1, cutoff=0.5)
#         if close_matches:
#             diffs.append(f"- {line}")
#             diffs.append(f"+ {close_matches[0]}")
#         else:
#             diffs.append(f"- {line}")
#     
#     for line in unique_to_text2:
#         if line not in [match[2:] for match in diffs if match.startswith("+")]:
#             diffs.append(f"+ {line}")
# 
#     return "\n".join(diffs) if diffs else "No significant differences."
# 
# 
# # Path to the directories containing metadata
# path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
# 
# 
# # Iterate over similar pairs and fetch their metadata
# for pair in similar_pairs:
#     sample_id1, sample_id2, similarity_score = pair
#     metadata1 = fetch_metadata_from_sample(sample_id1, path_to_dirs)
#     metadata2 = fetch_metadata_from_sample(sample_id2, path_to_dirs)
#     
#     print(f"Similarity Score: {similarity_score}")
#     # print(f"Metadata for {sample_id1}:\n{metadata1}\n")
#     # print(f"Metadata for {sample_id2}:\n{metadata2}\n")
#     
#     # Highlight and print the differences
#     differences = highlight_differences(metadata1, metadata2)
#     print("Highlighted Differences:\n")
#     print(differences)
#     print("--------------------------------------------------\n")
# =============================================================================


