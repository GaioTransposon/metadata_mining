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



# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Function to get embeddings from OpenAI
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
    return np.array([embedding['embedding'] for embedding in response['data']])

# Set your OpenAI API key
api_key_path ='/Users/dgaio/my_api_key'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()

categories = ["animal", "plant", "water", "soil"]

# Define colors for categories
category_colors = {"animal": "pink", "plant": "green", "water": "blue", "soil": "brown"}


# Path to your file
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb100_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240115_1716.txt'


# Read file and extract summaries and IDs
texts = []
sample_ids = []
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        sample_ids.append(row[0])  # Store Sample ID
        texts.append(row[1])       # Store Sample Text


# Get embeddings
category_embeddings = get_embeddings(categories)
text_embeddings = get_embeddings(texts)

# Plotting
pca = PCA(n_components=2)
all_embeddings = np.vstack((category_embeddings, text_embeddings))
transformed_embeddings = pca.fit_transform(all_embeddings)


# DataFrame to store results
results = []

# Function to normalize and scale values between a min and max range
def scale_normalize(values, min_scale=0.2, max_scale=1.0):
    min_val = min(values)
    max_val = max(values)
    return [min_scale + (max_scale - min_scale) * ((val - min_val) / (max_val - min_val) if max_val - min_val else 0) for val in values]

# Plot sample embeddings with assigned category colors and intensity based on similarity
for i, (sample_id, text, text_emb) in enumerate(zip(sample_ids, texts, text_embeddings)):
    similarities = [cosine_similarity(text_emb, cat_emb) for cat_emb in category_embeddings]
    assigned_category = categories[np.argmax(similarities)]
    scaled_similarities = scale_normalize(similarities)
    max_similarity = max(scaled_similarities)
    similarity_scores = {f"Similarity to {cat}": sim for cat, sim in zip(categories, similarities)}
    results.append({
        "Sample ID": sample_id,
        "Sample Text": text,
        "Predicted Category": assigned_category,
        **similarity_scores
    })
    color = category_colors[assigned_category]
    plt.scatter(*transformed_embeddings[i+len(categories)], color=color, alpha=max_similarity)

# Create legend entries for categories
legend_entries = [plt.Line2D([0], [0], marker='o', color='w', label=category, 
                             markerfacecolor=category_colors[category], markersize=10, markeredgewidth=2, markeredgecolor='black') for category in categories]

# Plot category embeddings with a thick contour and add the legend
for i, category in enumerate(categories):
    plt.scatter(*transformed_embeddings[i], color=category_colors[category], edgecolors='black', linewidths=2)

plt.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Text Embeddings Visualization')
plt.tight_layout()
plt.show()


# Create DataFrame
df = pd.DataFrame(results)

# Display the DataFrame
print(df)
