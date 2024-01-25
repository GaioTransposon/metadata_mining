#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:23:27 2024

@author: dgaio
"""


import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
import pickle
import umap
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import json
from annoy import AnnoyIndex



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


# compute cosine similarity
def custom_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


# rescale opacities
def rescale_opacity(opacities, new_min=0.3, new_max=1.0):
    old_min, old_max = min(opacities), max(opacities)
    return [new_min + (new_max - new_min) * ((opacity - old_min) / (old_max - old_min)) for opacity in opacities]


def calculate_max_similarity(annoy_index, embedding_index, n_neighbors=10):
    ids, distances = annoy_index.get_nns_by_item(embedding_index, n_neighbors, include_distances=True)
    cosine_similarities = [1 - distance ** 2 / 2 for distance in distances]
    return max(cosine_similarities)

################################


input_gold_dict = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"

with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict = gold_dict[0]

# Convert gold_dict to DataFrame
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['Sample ID', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)


################################


embeddings_file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings.json"

# Load embeddings and get sample IDs
sample_ids, my_embeddings = load_embeddings(embeddings_file_path)
print(f"The number of entries in the embeddings file is: {len(my_embeddings)}")

# Stack embeddings
all_embeddings = np.vstack(my_embeddings)

################################################################################
################################################################################



# Determine the size of your embeddings
embedding_size = my_embeddings[0].shape[0]


# Create an Annoy index
annoy_index = AnnoyIndex(embedding_size, 'angular')  # 'angular' distance is equivalent to cosine similarity

# Add all embeddings to the Annoy index
for i, embedding in enumerate(my_embeddings):
    annoy_index.add_item(i, embedding)

# Build the Annoy index
annoy_index.build(10)  # The number here is the number of trees in the index, more trees give higher precision


################################################################################
################################################################################



# K-means 

# Define your variables for coloring
color_by_cluster = 1  # Set to 1 for coloring by cluster, 0 otherwise
color_by_biome = 0    # Set to 1 for coloring by biome, 0 otherwise


# K-means Clustering
n_clusters = 10  # Change this number based on your expectation of distinct clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(all_embeddings)

# UMAP for dimensionality reduction
reducer = umap.UMAP()   # evt: random_state=42

transformed_embeddings = reducer.fit_transform(all_embeddings)

# Generate a list of 10 distinct colors
n_colors = 10  # Adjust this if you have more than 10 clusters
colors = plt.cm.get_cmap('tab10', n_colors)

# Convert colors to hex format for Plotly
hex_colors = [matplotlib.colors.rgb2hex(colors(i)) for i in range(n_colors)]




# Define biome colors
biome_colors = {"plant": "green", "water": "blue", "animal": "pink", "soil": "brown"}




# Convert the sample IDs from gold_dict_df to a set for efficient lookup
gold_dict_sample_ids = set(gold_dict_df['Sample ID'])

# Find common sample IDs
common_sample_ids = set(sample_ids) & gold_dict_sample_ids

# Check biome for common sample IDs
for sample_id in common_sample_ids:
    biome = gold_dict_df.loc[gold_dict_df['Sample ID'] == sample_id, 'biome'].values
    print(f"Sample ID: {sample_id}, Biome: {biome[0] if biome.size > 0 else 'Unknown'}")





# Initialize plot data and results
plot_data = []
results = []

# Define a thicker line width for highlighted samples
highlight_line_width = 2  # Adjust as needed

# Process each text embedding for plotting
for i, embedding in enumerate(transformed_embeddings):
    # Calculate the max similarity for the current embedding using Annoy
    max_similarity = calculate_max_similarity(annoy_index, i)

    # It's a text sample
    sample_id = sample_ids[i]
    

    # Check if the sample ID is in the common set and not 'Unknown'
    if sample_id in common_sample_ids:
        line_width = highlight_line_width
    else:
        line_width = None  # Default, no highlighting

    # Safely get biome from gold_dict_df
    biome = gold_dict_df.loc[gold_dict_df['Sample ID'] == sample_id, 'biome'].values

    # Determine color based on user selection
    if color_by_cluster:
        color = hex_colors[clusters[i]]
    elif color_by_biome:
        biome_color = biome_colors.get(biome[0], 'grey') if biome.size > 0 else 'grey'
        color = biome_color
    else:
        color = 'grey'  # Default color

    # Prepare text for hover information
    hover_text = f"Sample ID: {sample_id}<br>Biome: {biome[0] if biome.size > 0 else 'Unknown'}"

    # Add data to plot_data
    plot_data.append({
        "x": embedding[0],
        "y": embedding[1],
        "color": hex_colors[clusters[i]],  # Ensure correct color assignment
        "opacity": max_similarity,
        "text": hover_text,
        "line_width": line_width,
        "is_highlighted": sample_id in common_sample_ids  # Flag for highlighted points
    })

    # Populate your results DataFrame
    results.append({
        "Sample ID": sample_id,
        "Biome": biome[0] if biome.size > 0 else "Unknown",
        "Cluster": clusters[i],
    })
    
# Create a Plotly scatter plot
fig = go.Figure()

# Initialize a set to keep track of clusters already plotted
plotted_clusters = set()

# Plot non-highlighted points first
for data in plot_data:
    if not data['is_highlighted']:
        fig.add_trace(go.Scatter(
            x=[data['x']],
            y=[data['y']],
            mode='markers',
            marker=dict(color=data['color'], size=7, line=dict(width=data['line_width'], color='black')),
            text=data['text'],
            hoverinfo='text'
        ))

# Then plot highlighted points
for data in plot_data:
    if data['is_highlighted']:
        fig.add_trace(go.Scatter(
            x=[data['x']],
            y=[data['y']],
            mode='markers',
            marker=dict(color=data['color'], size=7, line=dict(width=data['line_width'], color='black')),
            text=data['text'],
            hoverinfo='text'
        ))


# Update layout with titles and labels
fig.update_layout(
    title='Text Embeddings Visualization with UMAP',
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2'
)

# Show the plot
fig.show()

# Save the figure as an HTML file
fig.write_html("text_embeddings_visualization.html")


# Create DataFrame for samples only
df = pd.DataFrame(results)

# Display the DataFrame
print(df)
    












