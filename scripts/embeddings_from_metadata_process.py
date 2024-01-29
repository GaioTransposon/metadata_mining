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
import json
from annoy import AnnoyIndex
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np




def load_embeddings(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            embeddings_dict = json.load(f)
        embeddings_list = [np.array(embeddings_dict[key]) for key in embeddings_dict]
        sample_ids = list(embeddings_dict.keys())
        return sample_ids, embeddings_list
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return [], []
    
    
def calculate_max_similarity(annoy_index, embedding_index, n_neighbors=10):
    ids, distances = annoy_index.get_nns_by_item(embedding_index, n_neighbors, include_distances=True)
    cosine_similarities = [1 - distance ** 2 / 2 for distance in distances]
    return max(cosine_similarities)



################################


# Load gold dictionary
input_gold_dict = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict = gold_dict[0]

# Convert gold_dict to DataFrame
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['Sample ID', 'Biome'])
gold_dict_df['Biome'] = gold_dict_df['Biome'].apply(lambda x: x[1])

# Load gold embeddings
embeddings_gd_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings_gd.json"
gd_sample_ids, gd_embeddings = load_embeddings(embeddings_gd_file_path)

# Create a DataFrame for gold embeddings and their biomes
gd_df = pd.DataFrame({'Sample ID': gd_sample_ids, 'Embedding': gd_embeddings})
gd_df = gd_df.merge(gold_dict_df, on='Sample ID', how='left')

# Randomly select 200 samples per biome
selected_gd_df = pd.concat([
    df.sample(n=min(200, len(df)), random_state=42) for _, df in gd_df.groupby('Biome')
])


# Load embeddings.json and filter out samples present in selected gold dict samples
embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings.json"
sample_ids, my_embeddings = load_embeddings(embeddings_file_path)
print(f"The number of entries in the embeddings file is: {len(my_embeddings)}")

# make sure embeddings from embeddings.json of samples that are present in gold dict, will be not be taken (we take those embeddings from embeddings_gd.pkl)
filtered_sample_ids = [sid for sid in sample_ids if sid not in selected_gd_df['Sample ID'].values]
len(filtered_sample_ids)

filtered_embeddings = [my_embeddings[sample_ids.index(sid)] for sid in filtered_sample_ids]
len(filtered_embeddings)

# Combine selected gold embeddings with the rest for analysis
combined_embeddings = np.vstack(filtered_embeddings + selected_gd_df['Embedding'].tolist())
len(combined_embeddings)







################################################################################



### Elbow method: 

sse = []
k_list = range(1, 20)  # Adjust the range of k as needed
for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(combined_embeddings)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_list, sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters k')
plt.ylabel('Sum of squared distances')
plt.show()




### Gaussian Mixture Method (on reduced data): 
    
# Dimensionality reduction with PCA
pca = PCA(n_components=50)  
reduced_embeddings = pca.fit_transform(combined_embeddings)

# Use a subset of data for faster computation
subset_indices = np.random.choice(reduced_embeddings.shape[0], size=1000, replace=False)  # Adjust size as needed
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

start_time = time.time()


# Proceed with clustering and UMAP
optimal_k = 10     # the number of clusters you found optimal to describe the data (via Gaussian or Elbow)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(combined_embeddings)
reducer = umap.UMAP(random_state=42)
transformed_embeddings = reducer.fit_transform(combined_embeddings)



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




# Visualization
fig = go.Figure()

# Plot non-gold dict samples
for i, (emb, cluster) in enumerate(zip(transformed_embeddings[:len(filtered_embeddings)], clusters[:len(filtered_embeddings)])):
    color = cluster_colors[cluster % len(cluster_colors)]  # Use modulo to cycle through cluster colors
    sample_id = filtered_sample_ids[i]
    # Retrieve biome for the sample ID, default to 'Unknown'
    biome = " "
    hover_text = f'Sample ID: {sample_id}<br>Biome: {biome}'
    fig.add_trace(go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers', marker=dict(color=color, size=7), text=hover_text, hoverinfo='text', showlegend=False))

# Plot gold dict samples with biome-specific colors
for i, emb in enumerate(transformed_embeddings[len(filtered_embeddings):]):
    sample_id = selected_gd_df.iloc[i]["Sample ID"]
    biome = selected_gd_df.iloc[i]["Biome"]
    # Ensure color is fetched correctly from biome_colors, default to 'gray'    
    color = biome_colors.get(biome, "white")
    hover_text = f'Sample ID: {sample_id}<br>Biome: {biome}'
    fig.add_trace(go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers', marker=dict(color=color, size=7, line=dict(width=2, color='black')), text=hover_text, hoverinfo='text', showlegend=True))

# Update layout
fig.update_layout(title='UMAP Projection of Embeddings', xaxis_title='UMAP 1', yaxis_title='UMAP 2', showlegend=True)

# Show plot
fig.show()

# Save the figure as an HTML file
fig.write_html("text_embeddings_visualization.html")



end_time = time.time()

z = (end_time - start_time) 
print(f"time taken: {z}")


    
    
################################################################################
################################################################################




































