#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:46:00 2024

@author: dgaio
"""


import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
METADATA_DIRECTORY = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
OUTPUT_FILE_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/top_similar_samples.txt"

# Load the existing data
with open(GOLD_DICT_PATH, 'rb') as file:
    data = pickle.load(file)

# Function to fetch metadata
def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(METADATA_DIRECTORY, folder_name)
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as file:
        return file.read()

# Collecting all metadata
all_metadata = {sample: fetch_metadata_from_sample(sample) for sample in data.keys()}
metadata_texts = list(all_metadata.values())
sample_ids = list(all_metadata.keys())

# Vectorizing the metadata texts
vectorizer = TfidfVectorizer()
metadata_vectors = vectorizer.fit_transform(metadata_texts)

# Calculating cosine similarities
similarity_matrix = cosine_similarity(metadata_vectors)

# Biome information extraction and color mapping for biomes
biomes = {sample: data[sample][1] for sample in data}
biome_colors = {
    'animal': '#C67D7B',  # pinkish
    'water': '#8CC8CF',   # bluish
    'plant': '#C0D184',   # greenish
    'soil': '#CBBF82',    # brownish
    'other': '#CCCCCC'    # gray
}

# Creating a DataFrame for pairwise sample similarities
rows = []
for idx in range(len(similarity_matrix)):
    for jdx in range(idx + 1, len(similarity_matrix)):  # Start from idx + 1 to avoid duplicates and self-comparisons
        if similarity_matrix[idx][jdx] >= 0.2:  # Filter by score
            same_biome = biomes[sample_ids[idx]] == biomes[sample_ids[jdx]]
            color_key = biomes[sample_ids[idx]] if same_biome else 'white'
            color = biome_colors.get(color_key, 'white')  # Default to white if no biome color found
            rows.append([sample_ids[idx], sample_ids[jdx], similarity_matrix[idx][jdx], color])

similarity_df = pd.DataFrame(rows, columns=['Sample_1', 'Sample_2', 'Similarity', 'Color'])

# Finding and saving the top n most similar samples
n = 20
top_similar_samples = similarity_df[similarity_df['Sample_1'] != similarity_df['Sample_2']].nlargest(n, 'Similarity')

with open(OUTPUT_FILE_PATH, 'w') as file:
    for _, row in top_similar_samples.iterrows():
        sample_1_metadata = fetch_metadata_from_sample(row['Sample_1'])
        sample_2_metadata = fetch_metadata_from_sample(row['Sample_2'])
        file_content = f"Metadata for '{row['Sample_1']}':\n{sample_1_metadata}\n\n"
        file_content += f"Metadata for '{row['Sample_2']}':\n{sample_2_metadata}\n"
        file_content += f"Similarity Score: {row['Similarity']}\n"
        file_content += "----------------------------------------\n\n"
        file.write(file_content)

print(f"Top {n} similar samples' metadata saved to {OUTPUT_FILE_PATH}")


+= 0.70 
# discard: 
SRS2619194


