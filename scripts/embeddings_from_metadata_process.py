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
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt



# Function to load embeddings from a .pkl file
def load_embeddings(file_path, gold_dict):
    try:
        with open(file_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        # Separate gold dict samples from others based on gold_dict keys
        gold_sample_ids = list(gold_dict.keys())
        gold_embeddings = []
        other_embeddings = []
        for sample_id, embedding in embeddings_data.items():
            if sample_id in gold_sample_ids:
                gold_embeddings.append((sample_id, np.array(embedding)))
            else:
                other_embeddings.append((sample_id, np.array(embedding)))
        
        return gold_embeddings, other_embeddings
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return [], []
    
    
def calculate_max_similarity(annoy_index, embedding_index, n_neighbors=10):
    ids, distances = annoy_index.get_nns_by_item(embedding_index, n_neighbors, include_distances=True)
    cosine_similarities = [1 - distance ** 2 / 2 for distance in distances]
    return max(cosine_similarities)



################################


# Load gold dictionary
input_gold_dict = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"  # Update this path
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict = gold_dict[0]

# Load embeddings from combined_data.pkl
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/combined_data.pkl"  # Update this path
gold_embeddings, other_embeddings = load_embeddings(embeddings_file_path, gold_dict)

# Optional: Subset the embeddings to a specific number if they exceed that number
max_embeddings = 50000  # Maximum number of embeddings to use
if len(other_embeddings) > max_embeddings:
    # Randomly select a subset of other_embeddings to reduce the size to max_embeddings
    # Ensure that this random selection is reproducible by setting a random seed
    np.random.seed(41)
    selected_indices = np.random.choice(len(other_embeddings), size=max_embeddings, replace=False)
    other_embeddings = [other_embeddings[i] for i in selected_indices]

# Convert lists of tuples to DataFrames
gold_df = pd.DataFrame(gold_embeddings, columns=['Sample ID', 'Embedding'])
other_df = pd.DataFrame(other_embeddings, columns=['Sample ID', 'Embedding'])


# Merge gold_df with gold_dict information to label biomes
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['Sample ID', 'Biome'])
gold_dict_df['Biome'] = gold_dict_df['Biome'].apply(lambda x: x[1])
gold_df = gold_df.merge(gold_dict_df, on='Sample ID', how='left')


# Randomly select 10 samples per biome from gold_df and reset the index
selected_gold_df = pd.concat([
    df.sample(n=min(100, len(df)), random_state=42) for _, df in gold_df.groupby('Biome')
]).reset_index(drop=True)

# Combine selected gold embeddings with the rest for analysis
combined_embeddings = np.vstack([emb for _, emb in other_df['Embedding'].iteritems()] + selected_gold_df['Embedding'].tolist())

len(combined_embeddings)


################################################################################


# How many clusters? 


### Gaussian Mixture Method (on reduced data): 
    
# Dimensionality reduction with PCA
pca = PCA(n_components=2)  
reduced_embeddings = pca.fit_transform(combined_embeddings)

# Use a subset of data for faster computation
subset_indices = np.random.choice(reduced_embeddings.shape[0], size=5000, replace=False)  # Adjust size as needed
reduced_embeddings_subset = reduced_embeddings[subset_indices]

# Fit GMMs on the reduced subset
n_components = np.arange(1, 11)  # Adjust range based on previous findings
models = [GaussianMixture(n, covariance_type='diag', random_state=42, max_iter=100).fit(reduced_embeddings_subset) for n in n_components]

bic_scores = [model.bic(reduced_embeddings_subset) for model in models]

plt.figure(figsize=(10, 6))
plt.plot(n_components, bic_scores, marker='o')
plt.title('BIC Score for Gaussian Mixture Models on Reduced Data')
plt.xlabel('Number of components')
plt.ylabel('BIC Score')
plt.show()


################################################################################


# Clustering:  

# Proceed with clustering and UMAP
optimal_k = 8     # the number of clusters you found optimal to describe the data (via Gaussian or Elbow)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# takes longish: 
clusters = kmeans.fit_predict(combined_embeddings)


###
# Print the unique cluster labels and their counts
unique_clusters, counts = np.unique(clusters, return_counts=True)
for cluster, count in zip(unique_clusters, counts):
    print(f"Cluster {cluster}: {count} samples")
    
print(np.unique(clusters[:len(other_embeddings)]))
###


# takes looooong: 
reducer = umap.UMAP(n_components=2, random_state=42)
transformed_embeddings = reducer.fit_transform(combined_embeddings)


################################################################################


# Plotting: 

start_time = time.time()



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
    # Add more colors if more than 10 clusters are needed
]



# Visualization enhancements
fig = go.Figure()

# Track which biomes and clusters have been added to legend to avoid duplication
added_to_legend_biomes = set()
added_to_legend_clusters = set()


def add_trace(embeddings, sample_ids, clusters=None, biomes=None, is_gold=False):
    for i, emb in enumerate(embeddings):
        sample_id = sample_ids[i]
        hover_text = f'Sample ID: {sample_id}'

        if is_gold:
            biome = biomes[i]
            color = biome_colors.get(biome, "white")  # Default to white if biome not found
            hover_text += f'<br>Biome: {biome}'
            show_in_legend = biome not in added_to_legend_biomes
            added_to_legend_biomes.add(biome)

            # Outline settings for gold samples
            marker_settings = dict(
                color=color,  # Fill color remains the same
                size=7,  # Size consistent with non-gold samples
                line=dict(color='black', width=2)  # Black outline with a width of 2
            )

        else:
            cluster = clusters[i]
            color = cluster_colors[cluster % len(cluster_colors)]
            show_in_legend = cluster not in added_to_legend_clusters
            added_to_legend_clusters.add(cluster)

            # Standard settings for non-gold samples, no outline
            marker_settings = dict(
                color=color,
                size=7,  # Consistent size with gold samples
                opacity=0.7  # Consistent opacity with gold samples (if needed)
            )


        # Adding trace with updated marker settings
        fig.add_trace(go.Scatter(
            x=[emb[0]], y=[emb[1]],
            mode='markers',
            marker=marker_settings,
            text=hover_text,
            hoverinfo='text',
            name=f"Biome: {biome}" if is_gold else f"Cluster: {cluster}",
            showlegend=show_in_legend
        ))



# Add non-gold samples to the plot
add_trace(transformed_embeddings[:len(other_embeddings)], [sid for sid, _ in other_embeddings], clusters[:len(other_embeddings)])

# Add gold samples to the plot
add_trace(transformed_embeddings[len(other_embeddings):], selected_gold_df['Sample ID'], biomes=selected_gold_df['Biome'], is_gold=True)



# Update layout for better visualization
fig.update_layout(
    title='UMAP Projection of Embeddings',
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2',
    legend_title_text='Legend',
    hovermode='closest'  # Update hover mode for better interaction
)

# Show plot
fig.show()


# Save the figure as an HTML file
fig.write_html("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/text_embeddings_visualization.html")


end_time = time.time()

z = (end_time - start_time) 
print(f"time taken to plot: {z}")


    
################################################################################
################################################################################


# Initialize UMAP with 3 components for 3D projection
reducer = umap.UMAP(n_components=3, random_state=42)

# Fit and transform the embeddings into 3D space
transformed_embeddings_3d = reducer.fit_transform(combined_embeddings)



# Initialize a new figure for 3D plotting
fig_3d = go.Figure()

# Non-gold samples with cluster-based coloring
non_gold_embeddings = transformed_embeddings_3d[:len(other_embeddings)]
non_gold_clusters = clusters[:len(other_embeddings)]
for cluster in np.unique(non_gold_clusters):
    cluster_indices = np.where(non_gold_clusters == cluster)[0]
    cluster_embeddings = non_gold_embeddings[cluster_indices]
    fig_3d.add_trace(go.Scatter3d(
        x=cluster_embeddings[:, 0],  # X coordinates
        y=cluster_embeddings[:, 1],  # Y coordinates
        z=cluster_embeddings[:, 2],  # Z coordinates
        mode='markers',
        marker=dict(
            size=5,  # Adjust size as needed
            color=cluster_colors[cluster % len(cluster_colors)],  # Use modular arithmetic for color selection
            opacity=0.7  # Adjust opacity as needed
        ),
        name=f'Cluster {cluster}'
    ))

# Gold samples with biome-based coloring
gold_embeddings = transformed_embeddings_3d[len(other_embeddings):]
gold_biomes = selected_gold_df['Biome'].values
unique_biomes = selected_gold_df['Biome'].unique()
for biome in unique_biomes:
    biome_indices = np.where(gold_biomes == biome)[0]
    biome_embeddings = gold_embeddings[biome_indices]
    fig_3d.add_trace(go.Scatter3d(
        x=biome_embeddings[:, 0],  # X coordinates
        y=biome_embeddings[:, 1],  # Y coordinates
        z=biome_embeddings[:, 2],  # Z coordinates
        mode='markers',
        marker=dict(
            size=5,  # Adjust size as needed
            color=biome_colors.get(biome, 'gray'),  # Default to gray if biome not found in the dictionary
            line=dict(
                color='black',  # Black outline for gold samples
                width=1  # Outline width
            ),
            opacity=0.7  # Adjust opacity as needed
        ),
        name=f'Biome: {biome}'
    ))

# Update layout for better visualization
fig_3d.update_layout(
    title='3D UMAP Projection of Embeddings with Cluster and Biome Colors',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    ),
    legend_title_text='Legend',
    legend=dict(
        itemsizing='constant'  # Ensure consistent legend marker size
    )
)

# Show the plot
fig_3d.show()


fig_3d.write_html("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/3d_text_embeddings_visualization.html")


################################################################################
################################################################################



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






################################











