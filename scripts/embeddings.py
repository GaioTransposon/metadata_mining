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
import matplotlib.pyplot as plt
import csv
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# compute cosine similarity
def cosine_similarity(vec1, vec2):
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

categories = ["animal", "plant", "water", "soil"]

category_descriptions = {
    "animal": "Animals including various species, from mammals to birds, living in different environments",
    "plant": "Diverse ecosystems of land plants including trees, flowers, and grasses",
    "water": "Aquatic environments ranging from oceans and seas to lakes and rivers",
    "soil": "Soil ecosystems comprising different types of soils, rich in minerals and organic matter"
}

# define colors for categories
category_colors = {"animal": "pink", "plant": "green", "water": "blue", "soil": "brown"}

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
#category_embeddings = get_embeddings(categories)
category_embeddings = get_embeddings(list(category_descriptions.values()))
text_embeddings = get_embeddings(texts)


# PCA
pca = PCA(n_components=2)
# vertically stacking category embeddings and text embeddings 
all_embeddings = np.vstack((category_embeddings, text_embeddings))
# apply PCA to all embeddings 
transformed_embeddings = pca.fit_transform(all_embeddings)


# initialization of empty lists 
plot_data = []
results = []  


# iterating over sample_ids, texts, and text_embeddings.
for i, (sample_id, text, text_emb) in enumerate(zip(sample_ids, texts, text_embeddings)):
    # compute embeddings 
    similarities = [cosine_similarity(text_emb, cat_emb) for cat_emb in category_embeddings]
    # maximum cosine similarity --> assume it's the samples biome 
    assigned_category = categories[np.argmax(similarities)]
    max_similarity = max(similarities) 
    color = category_colors[assigned_category]
    # prepping plot data: 
    plot_data.append({
        "x": transformed_embeddings[i+len(categories)][0],
        "y": transformed_embeddings[i+len(categories)][1],
        "text": f"ID: {sample_id}<br>Summary: {text[:200]}...",  # truncation of text to show
        "color": color,
        "opacity": max_similarity
    })
    results.append({"Sample ID": sample_id, "Sample Text": text, "Predicted Category": assigned_category, "Max Similarity": max_similarity})


# extract original opacities and rescale them
original_opacities = [data['opacity'] for data in plot_data]
rescaled_opacities = rescale_opacity(original_opacities)


# updating plot_data with rescaled opacities
for i, data in enumerate(plot_data):
    data['opacity'] = rescaled_opacities[i]
    

# initialize scatter plot
fig = go.Figure()

# category points added to scatter plot
for i, category in enumerate(categories):
    fig.add_trace(go.Scatter(x=[transformed_embeddings[i][0]], 
                             y=[transformed_embeddings[i][1]], 
                             mode='markers',
                             marker=dict(color=category_colors[category], size=10, line=dict(color='black', width=2)),
                             name=category))

# rescaled sample points added individually to scatter plot, to control their opacity
for data in plot_data:
    fig.add_trace(go.Scatter(x=[data['x']], 
                             y=[data['y']], 
                             mode='markers', 
                             marker=dict(color=data['color'], size=7, opacity=data['opacity']), 
                             text=data['text'],
                             hoverinfo='text',
                             showlegend=False))  # no legend needed for samples

fig.update_layout(title='Text Embeddings Visualization', xaxis_title='PCA 1', yaxis_title='PCA 2')
fig.show()

# save as html
fig.write_html("text_embeddings_visualization.html")

# save results in df format
df = pd.DataFrame(results)
print(df)




################################

# ALL embeddings against ALL 

# Compute similarities among all embeddings
all_embeddings = np.vstack((category_embeddings, text_embeddings))
similarity_matrix = cosine_similarity(all_embeddings)

# Initialize plot data and results
plot_data = []
results = []

# Process each embedding
for i, embedding in enumerate(all_embeddings):

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
        text_sample = texts[sample_index]
        assigned_category = 'Sample'
        color = 'grey'  # Or any other color to represent samples
        text = f"ID: {sample_id}<br>Summary: {text_sample[:200]}..." if len(text_sample) > 200 else f"ID: {sample_id}<br>Summary: {text_sample}"
        results.append({"Sample ID": sample_id, "Sample Text": text_sample, "Max Similarity": max_similarity})

    plot_data.append({
        "x": transformed_embeddings[i][0],
        "y": transformed_embeddings[i][1],
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
    title='Text Embeddings Visualization',
    xaxis_title='PCA Component 1',
    yaxis_title='PCA Component 2'
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





import pickle
import os

input_gold_dict = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"

with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict = gold_dict[0]

# Convert gold_dict to DataFrame
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['Sample ID', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)

# Merge with your existing DataFrame 'df'
merged_df = df.merge(gold_dict_df, on='Sample ID')


# Map biomes to colors for plotting
biome_colors = {biome: color for biome, color in zip(merged_df['biome'].unique(), ["brown", "blue", "green", "pink", "gray"])}  # Define your colors







