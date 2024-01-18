#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:55:10 2024

@author: dgaio
"""


import openai
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import csv
import plotly.graph_objects as go
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sklearn.cluster import KMeans



# compute cosine similarity
def custom_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# get embeddings from OpenAI
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
    return np.array([embedding['embedding'] for embedding in response['data']])

# rescale opacities
def rescale_opacity(opacities, new_min=0.3, new_max=1.0):
    old_min, old_max = min(opacities), max(opacities)
    return [new_min + (new_max - new_min) * ((opacity - old_min) / (old_max - old_min)) for opacity in opacities]

# open api key
api_key_path ='/Users/dgaio/my_api_key'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()


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


# path
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240117_1755.txt'

# read file and extract summaries and smaple IDs
texts = []
sample_ids = []
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        sample_ids.append(row[0])  # sample IDs
        texts.append(row[1])       # sample's summary 



# get embeddings
text_embeddings = get_embeddings(texts)

# Compute similarities among all embeddings
all_embeddings = np.vstack((text_embeddings))

similarity_matrix = cosine_similarity(all_embeddings)


# Define biome colors
biome_colors = {"plant": "green", "water": "blue", "animal": "pink", "soil": "brown"}

################################


# K-means 

# Define your variables for coloring
color_by_cluster = 0  # Set to 1 for coloring by cluster, 0 otherwise
color_by_biome = 1    # Set to 1 for coloring by biome, 0 otherwise


# K-means Clustering
n_clusters = 10  # Change this number based on your expectation of distinct clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(all_embeddings)

# UMAP for dimensionality reduction
reducer = umap.UMAP()   # evt: random_state=42

transformed_embeddings = reducer.fit_transform(all_embeddings)


# Define colors for K-means clusters (you can adjust these colors)
cluster_colors = ['red', 'green', 'blue', 'purple', 'orange'] * (n_clusters // 5 + 1)

# Initialize plot data and results
plot_data = []
results = []
# Process each text embedding for plotting
for i, embedding in enumerate(transformed_embeddings):
    # Calculate the max similarity for the current embedding
    max_similarity = max(similarity_matrix[i])

    # It's a text sample
    sample_id = sample_ids[i]
    
    # Safely get biome from gold_dict_df
    biome = gold_dict_df.loc[gold_dict_df['Sample ID'] == sample_id, 'biome'].values

    text_sample = texts[i]
    text = f"ID: {sample_id}<br>Summary: {text_sample[:200]}..." if len(text_sample) > 200 else f"ID: {sample_id}<br>Summary: {text_sample}"

    # Populate your results DataFrame
    results.append({"Sample ID": sample_id, "Biome": biome[0] if biome.size > 0 else "Unknown", "Cluster": clusters[i]})

    # Determine color based on user selection
    if color_by_cluster:
        color = cluster_colors[clusters[i]]
    elif color_by_biome:
        biome_color = biome_colors.get(biome[0], 'grey') if biome.size > 0 else 'grey'
        color = biome_color
    else:
        color = 'grey'  # Default color

    plot_data.append({
        "x": embedding[0],
        "y": embedding[1],
        "text": text,
        "color": color,
        "opacity": max_similarity
    })
    
# Create a Plotly scatter plot
fig = go.Figure()

# Add points to the scatter plot
for data in plot_data:
    fig.add_trace(go.Scatter(
        x=[data['x']], 
        y=[data['y']], 
        mode='markers', 
        marker=dict(color=data['color'], size=7), 
        text=data['text'],
        hoverinfo='text',
        showlegend=False  # Set showlegend to False
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

################################































