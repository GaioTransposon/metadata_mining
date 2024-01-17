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
from sklearn.decomposition import PCA
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



categories = ["animal", "plant", "water", "soil", "other"]

category_descriptions = {
    "animal": "Animals including various species, from mammals to birds, living in different environments",
    "plant": "Diverse ecosystems of land plants including trees, flowers, and grasses",
    "water": "Aquatic environments ranging from oceans and seas to lakes and rivers",
    "soil": "Soil ecosystems comprising different types of soils, rich in minerals and organic matter",
    "other": "laboratory, bioreactor, fungus, whole genome sequencing samples"
}

# define colors for categories
category_colors = {"animal": "pink", "plant": "green", "water": "blue", "soil": "brown", "other": "gray"}

# path
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb100_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240115_1716.txt'

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
category_embeddings = get_embeddings(list(category_descriptions.values()))
text_embeddings = get_embeddings(texts)

# Compute similarities among all embeddings
all_embeddings = np.vstack((category_embeddings, text_embeddings))

similarity_matrix = cosine_similarity(all_embeddings)


# Define biome colors
biome_colors = {"plant": "green", "water": "blue", "animal": "pink", "soil": "brown"}

################################



# K-means 


# K-means Clustering
n_clusters = 5  # Change this number based on your expectation of distinct clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(all_embeddings)

# UMAP for dimensionality reduction
reducer = umap.UMAP()
transformed_embeddings = reducer.fit_transform(all_embeddings)


# Define colors for K-means clusters (you can adjust these colors)
cluster_colors = ['red', 'green', 'blue', 'purple', 'orange'] * (n_clusters // 5 + 1)

# Initialize plot data and results
plot_data = []
results = []


# Process each embedding for plotting
for i, embedding in enumerate(transformed_embeddings):
    # Calculate the max similarity for the current embedding
    # Exclude the similarity with itself by setting diagonal to 0
    similarity_matrix[i, i] = 0
    max_similarity = max(similarity_matrix[i])
    
    
    # Determine if the current point is a category or a text sample
    if i < len(category_embeddings):
        # It's a category
        assigned_category = categories[i]
        color = category_colors[assigned_category]
        text = f"Category: {assigned_category}"
    else:
        # It's a text sample
        sample_index = i - len(category_embeddings)
        sample_id = sample_ids[sample_index]
        
        # Safely get biome from gold_dict_df
        biome = gold_dict_df.loc[gold_dict_df['Sample ID'] == sample_id, 'biome'].values
        biome_color = biome_colors.get(biome[0], 'grey') if biome.size > 0 else 'grey'

        text_sample = texts[sample_index]
        text = f"ID: {sample_id}<br>Summary: {text_sample[:200]}..." if len(text_sample) > 200 else f"ID: {sample_id}<br>Summary: {text_sample}"

        # Populate your results DataFrame
        results.append({"Sample ID": sample_id, "Biome": biome[0] if biome.size > 0 else "Unknown", "Cluster": clusters[sample_index]})

        # Use biome color for each sample
        color = biome_color

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































