#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:23:27 2024

@author: dgaio
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import time



# Load embeddings from a pickle file
def load_embeddings(file_path):
    """
    Load embeddings from a pickle file.

    Args:
        file_path (str): The file path to the pickle file containing the embeddings.

    Returns:
        list, list: A list of sample IDs and a list of corresponding embeddings.
    """
    try:
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)
        sample_ids = list(data_dict.keys())
        embeddings = [np.array(data_dict[key]) for key in data_dict]
        return sample_ids, embeddings
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return [], []




# Function to select N samples per biome
def select_n_samples_per_biome(gold_dict, combined_sample_ids, combined_embeddings, n=10):
    """
    Select N samples per biome from the gold dictionary within the combined embeddings.

    Args:
        gold_dict (dict): Gold dictionary with sample IDs as keys and biomes as values.
        combined_sample_ids (list): List of sample IDs in the combined embeddings.
        combined_embeddings (list): List of embeddings in the combined dataset.
        n (int): Number of samples to select per biome.

    Returns:
        list, list: Lists of selected sample IDs and their corresponding embeddings.
    """
    selected_embeddings = []
    selected_sample_ids = []
    biomes = pd.DataFrame(gold_dict.items(), columns=['Sample ID', 'Biome'])

    for biome in biomes['Biome'].unique():
        # Filter gold_dict for current biome and limit to N samples
        biome_sample_ids = biomes[biomes['Biome'] == biome]['Sample ID'].tolist()[:n]

        # Extract embeddings for selected samples
        for sample_id in biome_sample_ids:
            if sample_id in combined_sample_ids:
                index = combined_sample_ids.index(sample_id)
                selected_embeddings.append(combined_embeddings[index])
                selected_sample_ids.append(sample_id)

    return selected_sample_ids, selected_embeddings





# Load combined embeddings
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/temp/combined_data.pkl"  # Update this path
combined_sample_ids, combined_embeddings = load_embeddings(embeddings_file_path)

# Load gold dictionary
input_gold_dict = "/path/to/your/gold_dict.pkl"  # Update this path
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict = gold_dict[0]  # Assuming gold_dict is a dictionary with sample IDs as keys and biomes as values

# Select N samples per biome
n_samples_per_biome = 5  # Adjust N as needed
selected_sample_ids, selected_embeddings = select_n_samples_per_biome(gold_dict, combined_sample_ids, combined_embeddings, n=n_samples_per_biome)

# Ensure there's no duplication of selected samples in the remaining combined embeddings
remaining_embeddings = [emb for sid, emb in zip(combined_sample_ids, combined_embeddings) if sid not in selected_sample_ids]

# Combine selected gold dict embeddings with remaining embeddings for analysis
analysis_embeddings = np.vstack(selected_embeddings + remaining_embeddings)



################################################################################



### Elbow method: 

sse = []
k_list = range(1, 20)  # Adjust the range of k as needed
for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(analysis_embeddings)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_list, sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters k')
plt.ylabel('Sum of squared distances')
plt.show()



### Gaussian Mixture Method (on reduced data): 
    
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(analysis_embeddings)

subset_indices = np.random.choice(reduced_embeddings.shape[0], size=1000, replace=False)
reduced_embeddings_subset = reduced_embeddings[subset_indices]

n_components = np.arange(1, 11)
models = [GaussianMixture(n, covariance_type='diag', random_state=42, max_iter=100).fit(reduced_embeddings_subset) for n in n_components]

bic_scores = [model.bic(reduced_embeddings_subset) for model in models]

plt.figure(figsize=(10, 6))
plt.plot(n_components, bic_scores, marker='o')
plt.title('BIC Score for Gaussian Mixture Models on Reduced Data')
plt.xlabel('Number of components')
plt.ylabel('BIC Score')
plt.show()



################################################################################



### Clustering and UMAP
start_time = time.time()

optimal_k = 10  # the number of clusters you found optimal to describe the data (via Gaussian or Elbow)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(analysis_embeddings)
reducer = umap.UMAP(random_state=42)
transformed_embeddings = reducer.fit_transform(analysis_embeddings)

### Visualization
# Define biome colors
biome_colors = {
    "water": "blue",
    "plant": "green",
    "animal": "pink",
    "soil": "saddlebrown",
    "other": "gray"
}

# Define a list of hex colors for clusters, excluding the biome colors
cluster_colors = [
    '#ff7f0e',  # Orange
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink, different shade
    '#7f7f7f',  # Grey, different shade
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#1f77b4',  # Blue, different shade
    '#2ca02c',  # Green, different shade
]

fig = go.Figure()
added_to_legend = set()

for i, emb in enumerate(transformed_embeddings):
    # Determine if the sample is from the selected gold dict samples
    if i < len(selected_sample_ids):
        sample_id = selected_sample_ids[i]
        biome = gold_dict[sample_id]
        color = biome_colors.get(biome, "gray")  # Default to gray if biome not found
        show_in_legend = biome not in added_to_legend
        added_to_legend.add(biome)
        fig.add_trace(go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers',
                                 marker=dict(color=color, size=10, line=dict(width=2, color='black')),
                                 name=biome if show_in_legend else "",
                                 text=f'Sample ID: {sample_id}<br>Biome: {biome}',
                                 hoverinfo='text'))
    else:
        cluster = clusters[i - len(selected_sample_ids)]
        color = cluster_colors[cluster % len(cluster_colors)]
        fig.add_trace(go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers',
                                 marker=dict(color=color, size=7),
                                 hoverinfo='text',
                                 showlegend=False))

fig.update_layout(title='UMAP Projection of Embeddings', xaxis_title='UMAP 1', yaxis_title='UMAP 2')
fig.show()
fig.write_html("text_embeddings_visualization.html")

end_time = time.time()
print(f"time taken: {end_time - start_time}")






# =============================================================================
# 
# 
# start_time = time.time()
# 
# 
# # Proceed with clustering and UMAP
# optimal_k = 10     # the number of clusters you found optimal to describe the data (via Gaussian or Elbow)
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# clusters = kmeans.fit_predict(combined_embeddings)
# reducer = umap.UMAP(random_state=42)
# transformed_embeddings = reducer.fit_transform(combined_embeddings)
# 
# =============================================================================


# =============================================================================
# ## extra: 
# # Initialize Annoy index with Euclidean distance metric
# embedding_size = combined_embeddings.shape[1]  # Dimension of the embedding space
# annoy_index = AnnoyIndex(embedding_size, 'euclidean')  # Using 'euclidean' but 'angular' is also common
# 
# # Add embeddings to the Annoy index
# for i, embedding in enumerate(combined_embeddings):
#     annoy_index.add_item(i, embedding)
# 
# # Build the Annoy index
# annoy_index.build(10)  # The number of trees, more trees increase precision at the cost of performance
# ##
# =============================================================================

# =============================================================================
# 
# # Define biome colors
# biome_colors = {
#     "water": "blue",
#     "plant": "green",
#     "animal": "pink",
#     "soil": "saddlebrown",  
#     "other": "gray"
# }
# 
# 
# # Define a list of hex colors for clusters, excluding the biome colors
# cluster_colors = [
#     '#ff7f0e',  # Orange
#     '#d62728',  # Red
#     '#9467bd',  # Purple
#     '#8c564b',  # Brown
#     '#e377c2',  # Pink, different shade
#     '#7f7f7f',  # Grey, different shade
#     '#bcbd22',  # Olive
#     '#17becf',  # Cyan
#     '#1f77b4',  # Blue, different shade
#     '#2ca02c',  # Green, different shade
#     # Add more colors if more than 10 clusters are needed
# ]
# 
# 
# 
# # Visualization
# fig = go.Figure()
# 
# # Track which biomes have been added to legend
# added_to_legend = set()
# 
# 
# # Plot non-gold dict samples without adding to legend
# for i, (emb, cluster) in enumerate(zip(transformed_embeddings[:len(filtered_embeddings)], clusters[:len(filtered_embeddings)])):
#     color = cluster_colors[cluster % len(cluster_colors)]
#     sample_id = filtered_sample_ids[i]
#     hover_text = f'Sample ID: {sample_id}'
#     fig.add_trace(go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers',
#                              marker=dict(color=color, size=7),
#                              text=hover_text, hoverinfo='text',
#                              showlegend=False))  # Don't show these in legend
# 
# # Plot gold dict samples with biome-specific colors
# for i, emb in enumerate(transformed_embeddings[len(filtered_embeddings):]):
#     sample_id = selected_gd_df.iloc[i]["Sample ID"]
#     biome = selected_gd_df.iloc[i]["Biome"]
#     color = biome_colors.get(biome, "white")  # Use white as default if biome not found
#     hover_text = f'Sample ID: {sample_id}<br>Biome: {biome}'
# 
#     # Only add to legend if not already added
#     show_in_legend = biome not in added_to_legend
#     added_to_legend.add(biome)
# 
#     fig.add_trace(go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers',
#                              marker=dict(color=color, size=7, line=dict(width=2, color='black')),
#                              name=biome,  # This sets the legend name
#                              text=hover_text, hoverinfo='text',
#                              showlegend=show_in_legend))  # Show in legend only if not already added
# 
# 
# # Update layout
# fig.update_layout(title='UMAP Projection of Embeddings', xaxis_title='UMAP 1', yaxis_title='UMAP 2')
# 
# # Show plot
# fig.show()
# 
# # Save the figure as an HTML file
# fig.write_html("text_embeddings_visualization.html")
# 
# 
# 
# 
# end_time = time.time()
# 
# z = (end_time - start_time) 
# print(f"time taken: {z}")
# 
# =============================================================================

    
    
################################################################################
################################################################################




































