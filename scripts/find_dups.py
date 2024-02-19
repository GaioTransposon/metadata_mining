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
from annoy import AnnoyIndex
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances




# Path to the embeddings file
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/combined_data.pkl"

# Load embeddings and sample IDs from the file
with open(embeddings_file_path, 'rb') as file:
    data = pickle.load(file)
    
    
    
# Convert data to list and slice
sample_ids = list(data.keys())[:250000]  # Adjust the slice size as needed
all_embeddings = np.array(list(data.values()))[:250000]  # Slicing to match sample IDs




# part 1. split into minimum n clusters

def cluster_kmeans_exclude_singles(embeddings, sample_ids, n_clusters):
    start_time = time.time()

    # Apply k-means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=0, batch_size=1000)
    kmeans.fit(embeddings)

    # Initialize a dictionary to hold clusters
    clusters = {}

    # Group sample IDs based on cluster labels
    for idx, label in enumerate(kmeans.labels_):
        # Add sample ID to the corresponding cluster
        clusters.setdefault(label, []).append(sample_ids[idx])

    # Count the number of singletons before removing them
    singletons_count = sum(1 for members in clusters.values() if len(members) == 1)

    # Remove clusters that contain only one sample
    clusters = {label: members for label, members in clusters.items() if len(members) > 1}

    duration = time.time() - start_time

    # Return clusters, the number of singletons excluded, and the duration
    return clusters, singletons_count, duration


clusters, singletons_count, duration = cluster_kmeans_exclude_singles(all_embeddings, sample_ids, 100)
print(f"Number of singletons excluded: {singletons_count}")
print(f"Clustering duration: {duration}s")

# You can still list the clusters and get their count as before
list_of_clusters = list(clusters.values())
print(f"Number of clusters after excluding singletons: {len(list_of_clusters)}")

duration






# part 2. within each of the 100 clusters, split into further 10 clusters 


# Assume 'clusters' is a dictionary from the initial clustering, where each key is a cluster label
# and each value is a list of sample IDs in that cluster.

start_time = time.time()

# Initialize a list to store all final sub-clusters
final_sub_clusters = []

# Iterate over each initial cluster to further cluster its embeddings
for initial_cluster_label, member_ids in clusters.items():
    # Retrieve the embeddings for the current cluster's members
    member_embeddings = np.array([data[id] for id in member_ids])

    # Check if the current cluster has more than one member to avoid re-clustering single-member clusters
    if len(member_embeddings) > 1:
        # Apply sub-clustering to this subset
        sub_clusters, _ = cluster_kmeans_exclude_singles(member_embeddings, member_ids, 10)

        # Extend the list of final sub-clusters with the newly formed sub-clusters
        final_sub_clusters.extend(sub_clusters.values())

# The final_sub_clusters list now contains the sub-clusters formed within each of the initial clusters.
# Note that the total number of sub-clusters might be less than 1000 due to the exclusion of single-member clusters.

# Display the total number of sub-clusters obtained
total_sub_clusters = len(final_sub_clusters)
print(f"Total number of sub-clusters obtained: {total_sub_clusters}")


duration = time.time() - start_time
duration






# part 3. for each final sub-cluster, compute similarity matrix: 


def compute_similarity_for_cluster(sub_cluster_sample_ids, data, sub_cluster_index, similarity_threshold=0.99):
    start_time = time.time()  # Start timing

    # Extract the embeddings for the provided sample IDs
    sub_cluster_embeddings = np.array([data[id] for id in sub_cluster_sample_ids])

    # Calculate cosine similarity matrix for the subset
    similarity_matrix = cosine_similarity(sub_cluster_embeddings)

    # Initialize a graph
    G = nx.Graph()

    # Find pairs of similar embeddings based on the threshold and add edges to the graph
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] >= similarity_threshold:
                G.add_edge(sub_cluster_sample_ids[i], sub_cluster_sample_ids[j])

    # Find connected components, which represent groups of similar samples
    connected_components = list(nx.connected_components(G))

    # Create a dictionary to hold groups of similar samples
    groups = {f'sub_cluster_{sub_cluster_index}_group_{i + 1}': list(component) for i, component in enumerate(connected_components)}

    end_time = time.time()  # End timing
    print(f"Similarity computation for sub-cluster {sub_cluster_index} took {end_time - start_time:.2f} seconds.")

    return groups

len(final_sub_clusters)

# Apply the function to each sub-cluster
all_groups = []
for i, sub_cluster_sample_ids in enumerate(final_sub_clusters[0:100], start=1):
    print(f"Computing similarity for sub-cluster {i}...")
    groups = compute_similarity_for_cluster(sub_cluster_sample_ids, data, i)  # Pass the index 'i' as the sub-cluster identifier
    all_groups.append(groups)


len(all_groups)





total_number_of_groups = 0

for group_dict in all_groups:
    total_number_of_groups += len(group_dict)

# total number of embeddings incl in groups
total_embeddings_in_groups = sum(len(group) for group_dict in all_groups for group in group_dict.values())

# reduction of amples
reduction_in_samples = len(all_embeddings) - total_embeddings_in_groups + total_number_of_groups

# saved by clustering
embeddings_saved = len(all_embeddings) - reduction_in_samples


print("Clustering Summary:")
print("-" * 50)  
print(f"Initial number of samples: {len(all_embeddings)}")
print(f"Number of groups formed: {total_number_of_groups}")
print(f"Total embeddings included in groups: {total_embeddings_in_groups}")
print(f"Effective number of samples after grouping: {reduction_in_samples}")
print(f"Number of sample embeddings saved by clustering: {embeddings_saved}")
print("-" * 50) 
print("This reduction accounts for consolidating similar embeddings into groups, effectively reducing the number of distinct samples to consider.")














# =============================================================================
# # 1. Cosine Similarity with Graphs
# 
# start = time.time()
# 
# # Calculate cosine similarity matrix for the subset
# similarity_matrix = cosine_similarity(all_embeddings)
# 
# # Set a threshold for considering embeddings as 'nearly identical'
# similarity_threshold = 0.99
# 
# # Find pairs of similar embeddings based on the threshold
# similar_pairs = []
# for i in range(len(similarity_matrix)):
#     for j in range(i+1, len(similarity_matrix)):  # Compare each pair only once
#         if similarity_matrix[i, j] >= similarity_threshold:
#             similar_pairs.append((sample_ids[i], sample_ids[j], similarity_matrix[i, j]))
# 
# # Create a graph
# G = nx.Graph()
# 
# # Add edges for pairs with high similarity
# for sample1, sample2, score in similar_pairs:
#     if score >= 0.99:  # similarity threshold
#         G.add_edge(sample1, sample2)
# 
# # Find connected components, which represent groups of similar samples
# connected_components = list(nx.connected_components(G))
# 
# # Create a dictionary to hold groups of similar samples
# groups = {}
# for i, component in enumerate(connected_components):
#     group_name = f'group_{i + 1}'
#     groups[group_name] = list(component)
# 
# # Print the groups
# for group_name, samples in groups.items():
#     pass
#     #print(f"{group_name}: {samples}")
# 
# len(groups)
# 
# end = time.time()
# print(end-start)
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










