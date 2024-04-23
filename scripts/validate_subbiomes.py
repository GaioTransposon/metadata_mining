#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""

import os
import pickle
import csv
import openai
import time
from datetime import datetime
import numpy as np
from scipy.spatial import distance
import plotly.graph_objects as go
import umap


def make_biome_subbiome_dict(source, source_type='csv'):
    result_dict = {}
    if source_type == 'pickle':
        # Load data from a pickle file
        with open(source, 'rb') as file:
            data, processed_pmids = pickle.load(file)
        # Extracting data from the dictionary
        for key, values in data.items():
            if len(values) >= 3:
                result_dict[key] = f"{values[1]} - {values[2]}"
    elif source_type == 'csv':
        # Load data from a CSV file
        try:
            with open(source, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    result_dict[row['col_0']] = f"{row['col_1']} - {row['col_4']}"
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    return result_dict



def get_embeddings(data_dict, verbose='no'):
    embeddings_dict = {}
    failed_samples = []

    # Prepare data for API call
    sample_ids = list(data_dict.keys())
    descriptions = list(data_dict.values())

    try:
        start_api_call_time = time.time()

        # Process in batches to manage API call size and rate limits
        batch_size = 20  # Define batch size, adjust based on API limits
        for i in range(0, len(descriptions), batch_size):
            chunk = descriptions[i:i + batch_size]
            sample_ids_chunk = sample_ids[i:i + batch_size]

            # Make the API call using OpenAI's client library
            try:
                response = openai.Embedding.create(input=chunk, engine="text-embedding-ada-002")
                embeddings = [embedding['embedding'] for embedding in response['data']]
                for j, sample_id in enumerate(sample_ids_chunk):
                    embeddings_dict[sample_id] = np.array(embeddings[j], dtype=np.float32)
            except openai.error.OpenAIError as e:
                print(f"API error for batch starting with {sample_ids_chunk[0]}: {e}")
                failed_samples.extend(sample_ids_chunk)
            except Exception as e:
                print(f"Unexpected error for batch starting with {sample_ids_chunk[0]}: {e}")
                failed_samples.extend(sample_ids_chunk)

        end_api_call_time = time.time()

        if verbose.lower() == 'yes':
            print(f"{datetime.now()} - API call for {len(descriptions)} texts took {end_api_call_time - start_api_call_time:.2f} seconds")

    except Exception as e:
        print(f"General error in embedding function: {e}")
        # Mark all remaining as failed
        failed_samples.extend(sample_ids[i:])

    return embeddings_dict, failed_samples


def plot_embeddings_1D_to_html(embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2):
    x, y, hover_texts, colors = [], [], [], []

    # Define specific hex colors for each biome
    color_map = {
        'animal': '#C67D7B',  # Pinkish
        'water': '#8CC8CF',   # Bluish
        'plant': '#C0D184',   # Greenish
        'soil': '#CBBF82',    # Brownish
        'other': '#CCCCCC'    # Gray
    }

    for sample_id in embeddings_dict1:
        if sample_id in embeddings_dict2 and sample_id in labels_dict1 and sample_id in labels_dict2:
            vec1 = embeddings_dict1[sample_id]
            vec2 = embeddings_dict2[sample_id]
            if vec1 is not None and vec2 is not None:
                print(vec1[1])
                x.append(vec1[1])  # Simplified projection to the first dimension
                y.append(vec2[1])  # Simplified projection to the first dimension
                biome = labels_dict1[sample_id].split(' - ')[0]  # Extract biome part
                colors.append(color_map.get(biome, '#CCCCCC'))  # Use gray for unknown biomes
                hover_text = f"'{sample_id}':<br>gold_dict: '{labels_dict1[sample_id]}'<br>gpt: '{labels_dict2[sample_id]}'"
                hover_texts.append(hover_text)

    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', text=hover_texts, hoverinfo='text', marker=dict(color=colors)))
    fig.update_layout(title='Comparison of Embeddings',
                      xaxis_title='GPT Embeddings',
                      yaxis_title='Gold Dict Embeddings',
                      hovermode='closest')
    fig.write_html("/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes.html")


def plot_reduced_embeddings_to_html(embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2):
    embeddings1 = np.array([embeddings_dict1[sample] for sample in embeddings_dict1 if sample in embeddings_dict2])
    embeddings2 = np.array([embeddings_dict2[sample] for sample in embeddings_dict2 if sample in embeddings_dict1])

    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embeddings1_reduced = reducer.fit_transform(embeddings1)
    embeddings2_reduced = reducer.fit_transform(embeddings2)

    x = embeddings1_reduced[:, 0]  # First component from UMAP
    y = embeddings2_reduced[:, 0]  # First component from UMAP for the second set of embeddings

    hover_texts = []
    colors = []

    # Define specific hex colors for each biome
    color_map = {
        'animal': '#C67D7B',  # Pinkish
        'water': '#8CC8CF',   # Bluish
        'plant': '#C0D184',   # Greenish
        'soil': '#CBBF82',    # Brownish
        'other': '#CCCCCC'    # Gray
    }

    for i, sample_id in enumerate(embeddings_dict1):
        if sample_id in embeddings_dict2:
            biome = labels_dict1[sample_id].split(' - ')[0]
            colors.append(color_map.get(biome, '#CCCCCC'))
            hover_text = f"'{sample_id}':<br>gold_dict: '{labels_dict1[sample_id]}'<br>gpt: '{labels_dict2[sample_id]}'"
            hover_texts.append(hover_text)

    # Create scatter plot
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', text=hover_texts, hoverinfo='text', marker=dict(color=colors)))
    fig.update_layout(title='Comparison of Embeddings via UMAP',
                      xaxis_title='GPT Embeddings UMAP',
                      yaxis_title='Gold Dict Embeddings UMAP',
                      hovermode='closest')
    fig.write_html("/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes_umap.html")






# STEP 1. Extract biome and sub-biome from gold dict and/or from gpt output


# Example usage for pickle file
GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
gold_dict_bsb = make_biome_subbiome_dict(GOLD_DICT_PATH, source_type='pickle')
print(gold_dict_bsb)

# Example usage for CSV file
CSV_PATH = '/Users/dgaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs32_API226_normal_dt20240415_1859.txt'
gpt_clean_bsb = make_biome_subbiome_dict(CSV_PATH, source_type='csv')
print(gpt_clean_bsb)



# STEP 2. reduce to keep common samples


# Calculate the intersection of keys from both dictionaries
common_keys = set(gold_dict_bsb.keys()) & set(gpt_clean_bsb.keys())

# Filter both dictionaries to only include common keys
gold_dict_bsb_filtered = {k: v for k, v in gold_dict_bsb.items() if k in common_keys}
gpt_clean_bsb_filtered = {k: v for k, v in gpt_clean_bsb.items() if k in common_keys}

# Print filtered dictionaries
print("Filtered gold_dict_bsb:")
print(gold_dict_bsb_filtered)
len(gold_dict_bsb_filtered)

print("Filtered gpt_clean_bsb:")
print(gpt_clean_bsb_filtered)
len(gpt_clean_bsb_filtered)



# STEP 3. get embeddings

# Initialize API key
with open('/Users/dgaio/my_api_key', "r") as file:
    openai.api_key = file.read().strip()
    
embeddings_gpt, failed_samples_gpt = get_embeddings(gpt_clean_bsb_filtered, verbose='yes')
len(embeddings_gpt)
len(failed_samples_gpt)
embeddings_gd, failed_samples_gd = get_embeddings(gold_dict_bsb_filtered, verbose='yes')
len(embeddings_gd)
len(failed_samples_gd)

print("Embeddings Dictionary:")
for sample, embedding in embeddings_gpt.items():
    print(f"{sample}: {embedding[:5]}")  # Print first 5 elements for brevity



# STEP 4. plot embeddings 

plot_embeddings_1D_to_html(embeddings_gpt, embeddings_gd, gold_dict_bsb_filtered, gpt_clean_bsb_filtered)

plot_reduced_embeddings_to_html(embeddings_gpt, embeddings_gd, gold_dict_bsb_filtered, gpt_clean_bsb_filtered)





