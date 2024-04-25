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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


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



def plot_embeddings_1D_to_html(embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2, X):
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
                x.append(vec1[X])  # Simplified projection to the first dimension
                y.append(vec2[X])  # Simplified projection to the first dimension
                biome = labels_dict1[sample_id].split(' - ')[0]  # Extract biome part
                colors.append(color_map.get(biome, '#CCCCCC'))  # Use gray for unknown biomes
                hover_text = f"'{sample_id}':<br>gold_dict: '{labels_dict1[sample_id]}'<br>gpt: '{labels_dict2[sample_id]}'"
                hover_texts.append(hover_text)

    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', text=hover_texts, hoverinfo='text', marker=dict(color=colors)))
    fig.update_layout(title='Comparison of Embeddings',
                      xaxis_title='GPT Embeddings',
                      yaxis_title='Gold Dict Embeddings',
                      hovermode='closest')
    fig.write_html(f"/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes_dim{X}.html")



# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
# 
# # Extracting all embeddings into a list
# all_embeddings = list(embeddings_gd.values())
# 
# all_embeddings_array = np.array(all_embeddings)
# 
# # Check the shape to ensure it's correct (should be number of samples x embedding dimensions)
# print(all_embeddings_array.shape)
# 
# 
# 
# # Calculate the average absolute value and variance for each dimension
# avg_abs_values = np.mean(np.abs(all_embeddings), axis=0)
# variances = np.var(all_embeddings, axis=0)
# 
# # Find the indices of the top 6 dimensions with the highest average absolute values
# top_indices = np.argsort(-avg_abs_values)[:6]  # negating to sort in descending order
# 
# 
# # Plot the results
# fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# 
# axs[0].plot(avg_abs_values)
# axs[0].set_title('Average Absolute Value of Embedding Dimensions')
# axs[0].set_xlabel('Dimension')
# axs[0].set_ylabel('Average Absolute Value')
# 
# axs[1].plot(variances)
# axs[1].set_title('Variance of Embedding Dimensions')
# axs[1].set_xlabel('Dimension')
# axs[1].set_ylabel('Variance')
# 
# plt.tight_layout()
# plt.show()
# 
# 
# # Print the top 6 indices
# print("Top 6 indices with the highest average absolute values:", top_indices)
# =============================================================================




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
with open('/Users/dgaio/my_api_key_embeddings', "r") as file:
    openai.api_key = file.read().strip()
    
embeddings_gpt, failed_samples_gpt = get_embeddings(gpt_clean_bsb_filtered, verbose='yes')
print(len(embeddings_gpt))
print(len(failed_samples_gpt))
embeddings_gd, failed_samples_gd = get_embeddings(gold_dict_bsb_filtered, verbose='yes')
print(len(embeddings_gd))
print(len(failed_samples_gd))

print("Embeddings Dictionary:")
for sample, embedding in embeddings_gd.items():
    print(f"{sample}: {embedding[:5]}")  # Print first 5 elements for brevity



# STEP 4. plot embeddings 


# Top 6 indices with the highest average absolute values: [ 194  954 1120 1246 1348 1487] all minus 1! 
plot_embeddings_1D_to_html(embeddings_gd, embeddings_gpt, gold_dict_bsb_filtered, gpt_clean_bsb_filtered, 1486)





# STEP 5. Computing similarity: 





def compare_embeddings(embeddings_dict1, embeddings_dict2):
    comparison_results = {}
    for sample_id, embedding1 in embeddings_dict1.items():
        if sample_id in embeddings_dict2:
            embedding2 = embeddings_dict2[sample_id]
            euclidean_dist = distance.euclidean(embedding1, embedding2)
            cosine_sim = distance.cosine(embedding1, embedding2)
            manhattan_dist = distance.cityblock(embedding1, embedding2)
            comparison_results[sample_id] = {
                'euclidean': euclidean_dist,
                'cosine': 1 - cosine_sim,  # since scipy's cosine returns 1 - similarity
                'manhattan': manhattan_dist
            }
    return comparison_results



# Assuming 'compare_results' is the dictionary returned from the 'compare_embeddings' function
def plot_distribution_metrics(compare_results):
    euclidean_distances = [result['euclidean'] for result in compare_results.values()]
    cosine_similarities = [result['cosine'] for result in compare_results.values()]
    manhattan_distances = [result['manhattan'] for result in compare_results.values()]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plotting Euclidean Distances
    sns.histplot(euclidean_distances, bins=30, kde=True, ax=axs[0])
    axs[0].set_title('Euclidean Distance Distribution')
    axs[0].set_xlabel('Euclidean Distance')
    axs[0].set_ylabel('Frequency')

    # Plotting Cosine Similarities
    sns.histplot(cosine_similarities, bins=30, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Cosine Similarity Distribution')
    axs[1].set_xlabel('Cosine Similarity')
    axs[1].set_ylabel('Frequency')

    # Plotting Manhattan Distances
    sns.histplot(manhattan_distances, bins=30, kde=True, ax=axs[2], color='red')
    axs[2].set_title('Manhattan Distance Distribution')
    axs[2].set_xlabel('Manhattan Distance')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()




compare_embeddings(embeddings_gd, embeddings_gpt)


compare_results = compare_embeddings(embeddings_gd, embeddings_gpt)  # Make sure you have these dictionaries ready
plot_distribution_metrics(compare_results)






import numpy as np
import plotly.graph_objects as go

def plot_distribution_with_improved_details(compare_results, embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2):
    # Extract metrics
    euclidean_distances = [result['euclidean'] for result in compare_results.values()]
    sample_ids = list(compare_results.keys())
    
    # Prepare color mapping and hover texts
    color_map = {
        'animal': '#C67D7B',  # Pinkish
        'water': '#8CC8CF',   # Bluish
        'plant': '#C0D184',   # Greenish
        'soil': '#CBBF82',    # Brownish
        'other': '#CCCCCC'    # Gray
    }
    colors = [color_map.get(labels_dict1[key].split(' - ')[0], '#CCCCCC') for key in sample_ids]
    hover_texts = [f"{key}:<br>Gold: {labels_dict1[key]}<br>GPT: {labels_dict2[key]}" for key in sample_ids]

    # Create histogram and get bin information
    bin_size = np.ptp(euclidean_distances) / 30  # Define the bin size
    fig = go.Figure()
    hist = np.histogram(euclidean_distances, bins=np.arange(min(euclidean_distances), max(euclidean_distances) + bin_size, bin_size))
    bin_edges = hist[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create the histogram
    fig.add_trace(go.Histogram(x=euclidean_distances, xbins=dict(start=min(euclidean_distances), end=max(euclidean_distances), size=bin_size), opacity=0.75, name='Euclidean Distance'))

    # Calculate positions for scatter points
    y_positions = {}
    for dist in euclidean_distances:
        bin_index = int((dist - min(euclidean_distances)) / bin_size)
        if bin_index not in y_positions:
            y_positions[bin_index] = 0
        y_positions[bin_index] += 1
    
    scatter_x = []
    scatter_y = []
    for i, dist in enumerate(euclidean_distances):
        bin_index = int((dist - min(euclidean_distances)) / bin_size)
        scatter_x.append(dist)
        scatter_y.append(y_positions[bin_index])

    # Create scatter plot on top of the histogram
    fig.add_trace(go.Scatter(x=scatter_x, y=scatter_y, mode='markers', text=hover_texts, hoverinfo='text', marker=dict(color=colors, size=7)))

    # Update layout
    fig.update_layout(title='Euclidean Distance Distribution with Sample Details',
                      xaxis_title='Euclidean Distance',
                      yaxis_title='Sample Spread',
                      hovermode='closest',
                      showlegend=False)
    fig.show()
    fig.write_html(f"/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes_similarity.html")

# Example usage
compare_results = compare_embeddings(embeddings_gd, embeddings_gpt)  # Make sure you have these dictionaries ready
plot_distribution_with_improved_details(compare_results, embeddings_gd, embeddings_gpt, gold_dict_bsb_filtered, gpt_clean_bsb_filtered)





import numpy as np
import plotly.graph_objects as go

def plot_distribution_with_spread_details(compare_results, embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2):
    # Extract metrics
    euclidean_distances = [result['euclidean'] for result in compare_results.values()]
    sample_ids = list(compare_results.keys())

    # Prepare color mapping and hover texts
    color_map = {
        'animal': '#C67D7B',  # Pinkish
        'water': '#8CC8CF',   # Bluish
        'plant': '#C0D184',   # Greenish
        'soil': '#CBBF82',    # Brownish
        'other': '#CCCCCC'    # Gray
    }
    colors = [color_map.get(labels_dict1[key].split(' - ')[0], '#CCCCCC') for key in sample_ids]
    hover_texts = [f"{key}:<br>Gold: {labels_dict1[key]}<br>GPT: {labels_dict2[key]}" for key in sample_ids]

    # Create histogram and get bin information
    bin_size = np.ptp(euclidean_distances) / 30  # Define the bin size
    fig = go.Figure()
    hist = np.histogram(euclidean_distances, bins=np.arange(min(euclidean_distances), max(euclidean_distances) + bin_size, bin_size))
    bin_edges = hist[1]
    bin_counts = hist[0]

    # Calculate positions for scatter points
    y_positions = {}
    scatter_x = []
    scatter_y = []

    for dist in euclidean_distances:
        bin_index = int((dist - min(euclidean_distances)) / bin_size)
        if bin_index not in y_positions:
            y_positions[bin_index] = []
        y_positions[bin_index].append(dist)

    for bin_index, dists in y_positions.items():
        heights = np.linspace(1, bin_counts[bin_index], num=len(dists))  # Evenly spread out within the bin
        scatter_x.extend(dists)
        scatter_y.extend(heights)

    # Create the histogram
    fig.add_trace(go.Histogram(x=euclidean_distances, xbins=dict(start=min(euclidean_distances), end=max(euclidean_distances), size=bin_size), opacity=0.75, name='Euclidean Distance'))

    # Create scatter plot on top of the histogram
    fig.add_trace(go.Scatter(x=scatter_x, y=scatter_y, mode='markers', text=hover_texts, hoverinfo='text', marker=dict(color=colors, size=5)))

    # Update layout
    fig.update_layout(title='Euclidean Distance Distribution with Detailed Sample Spread',
                      xaxis_title='Euclidean Distance',
                      yaxis_title='Sample Position within Bin',
                      hovermode='closest',
                      showlegend=False)
    fig.show()
    fig.write_html(f"/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes_similarity.html")

# Example usage
compare_results = compare_embeddings(embeddings_gd, embeddings_gpt)  # Make sure you have these dictionaries ready
plot_distribution_with_spread_details(compare_results, embeddings_gd, embeddings_gpt, gold_dict_bsb_filtered, gpt_clean_bsb_filtered)




