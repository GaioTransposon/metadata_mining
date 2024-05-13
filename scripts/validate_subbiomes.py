#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""


# mind! needs version 0.28 !
# pip install openai==0.28

import pickle
import csv
import openai
import time
from datetime import datetime
import numpy as np
from scipy.spatial import distance
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics.pairwise import cosine_similarity

def make_biome_subbiome_dict(source, source_type='csv'):
    result_dict = {}
    
    # if gold dict, then load: 
    if source_type == 'pickle':
        with open(source, 'rb') as file:
            data = pickle.load(file)
        for key, values in data.items():
            if len(values) >= 3:
                result_dict[key] = f"{values[1]} - {values[2]}"
                
    # if gpt output (csv), then load: 
    elif source_type == 'csv':
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

    sample_ids = list(data_dict.keys())
    descriptions = list(data_dict.values())
    

    try:
        start_api_call_time = time.time()
        
        # process in batches to manage API call size and rate limits
        batch_size = 20
        for i in range(0, len(descriptions), batch_size):
            chunk = descriptions[i:i + batch_size]
            sample_ids_chunk = sample_ids[i:i + batch_size]

            # make API call
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

    # except Exception as e:
    #     print(f"General error in embedding function: {e}")
    #     failed_samples.extend(sample_ids[i:])
    
    except openai.APIError as e:
        print(f"API error for batch starting with {sample_ids_chunk[0]}: {e}")
        failed_samples.extend(sample_ids_chunk)
    except Exception as e:
        print(f"Unexpected error for batch starting with {sample_ids_chunk[0]}: {e}")
        failed_samples.extend(sample_ids_chunk)


    return embeddings_dict, failed_samples


# =============================================================================
# def plot_embeddings_1D_to_html(embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2, X):
#     x, y, hover_texts, colors = [], [], [], []
# 
#     # hex colors for each biome
#     color_map = {
#         'animal': '#C67D7B',  # pinkish
#         'water': '#8CC8CF',   # bluish
#         'plant': '#C0D184',   # greenish
#         'soil': '#CBBF82',    # brownish
#         'other': '#CCCCCC'    # gray
#     }
# 
#     for sample_id in embeddings_dict1:
#         if sample_id in embeddings_dict2 and sample_id in labels_dict1 and sample_id in labels_dict2:
#             vec1 = embeddings_dict1[sample_id]
#             vec2 = embeddings_dict2[sample_id]
#             if vec1 is not None and vec2 is not None:
#                 x.append(vec1[X])  # projection of just one dimension!
#                 y.append(vec2[X])  # projection of just one dimension!
#                 biome = labels_dict1[sample_id].split(' - ')[0]  # extract biome part to assign color
#                 colors.append(color_map.get(biome, '#000000'))  # Use black for unknown biomes
#                 hover_text = f"'{sample_id}':<br>gold_dict: '{labels_dict1[sample_id]}'<br>gpt: '{labels_dict2[sample_id]}'"
#                 hover_texts.append(hover_text)
# 
#     fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', text=hover_texts, hoverinfo='text', marker=dict(color=colors)))
#     fig.update_layout(title='Embeddings comparison',
#                       xaxis_title='GPT embeddings',
#                       yaxis_title='ground truth embeddings',
#                       hovermode='closest')
#     fig.write_html(f"/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes_dim{X}.html")
# =============================================================================


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


def plot_distribution_metrics(compare_results):
    euclidean_distances = [result['euclidean'] for result in compare_results.values()]
    cosine_similarities = [result['cosine'] for result in compare_results.values()]
    manhattan_distances = [result['manhattan'] for result in compare_results.values()]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.histplot(euclidean_distances, bins=30, kde=True, ax=axs[0])
    axs[0].set_title('Euclidean distance distribution')
    axs[0].set_xlabel('Euclidean distance')
    axs[0].set_ylabel('Frequency')

    sns.histplot(cosine_similarities, bins=30, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Cosine similarity distribution')
    axs[1].set_xlabel('Cosine similarity')
    axs[1].set_ylabel('Frequency')

    sns.histplot(manhattan_distances, bins=30, kde=True, ax=axs[2], color='red')
    axs[2].set_title('Manhattan distance distribution')
    axs[2].set_xlabel('Manhattan distance')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
    
    
def create_shuffled_background_distribution(embeddings_gd, embeddings_gpt, num_comparisons=None):
    random_cosine_similarities = []
    gd_keys = list(embeddings_gd.keys())
    gpt_keys = list(embeddings_gpt.keys())
    
    random.shuffle(gpt_keys)
    num_comparisons = min(len(gd_keys), len(gpt_keys))
    
    for i in range(num_comparisons):
        gd_embedding = embeddings_gd[gd_keys[i]]
        gpt_embedding = embeddings_gpt[gpt_keys[i]]
        cosine_sim = 1 - distance.cosine(gd_embedding, gpt_embedding)
        random_cosine_similarities.append(cosine_sim)
    
    return random_cosine_similarities


    
def plot_comparison_distribution(actual_similarities, background_similarities):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.histplot(actual_similarities, bins=30, kde=True, color='green', label='Cosine similarities', ax=ax, stat='density')
    
    sns.histplot(background_similarities, bins=30, kde=True, color='blue', label='Background cosine similarities', ax=ax, alpha=0.5, stat='density')
    
    ax.set_title('Actual vs Background similarities')
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('probability density')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    

def sample_by_category(labels, n):
    
    random.seed(42) 
    
    # split labels into biome and subbiome
    categorized = {k: v.split(' - ')[0] for k, v in labels.items()}
    
    category_groups = {}
    for key, category in categorized.items():
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(key)

    sampled_keys = []
    for category, keys in category_groups.items():
        if len(keys) > n:
            sampled_keys.extend(random.sample(keys, n))
        else:
            sampled_keys.extend(keys)  # if less than n, take all
    return sampled_keys


########################################


# STEP 1. Extract biome and sub-biome from gold dict and/or from gpt output


GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
gold_dict_bsb = make_biome_subbiome_dict(GOLD_DICT_PATH, source_type='pickle')
print(gold_dict_bsb)


#CSV_PATH = '/Users/dgaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API233_normal_dt20240503_1348.txt'
CSV_PATH = '/Users/dgaio/MicrobeAtlasProject/gpt_clean_output_modelgpt-3.5-turbo-1106_temp1.0_file-pwf3cm7lru1SPLZAf734bJ3Z_dt20240508_1655.txt'
gpt_clean_bsb = make_biome_subbiome_dict(CSV_PATH, source_type='csv')
print(gpt_clean_bsb)
len(gpt_clean_bsb)


########################################


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


########################################


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

print("Embeddings Dictionary (last 5 items):")
items = list(embeddings_gd.items())[-5:] 
for sample, embedding in items:
    print(f"{sample}: {embedding}")


########################################


# =============================================================================
# # STEP 3.1 (optional) Find out which dimensions of the embeddings are most informative: 
# 
# all_embeddings = list(embeddings_gpt.values())
# all_embeddings_array = np.array(all_embeddings)
# print(all_embeddings_array.shape)
# 
# # calculate average absolute value and variance for each dimension
# avg_abs_values = np.mean(np.abs(all_embeddings), axis=0)
# variances = np.var(all_embeddings, axis=0)
# 
# # Find indices of the top 6 dimensions with the highest average absolute values
# top_indices = np.argsort(-avg_abs_values)[:6]
# 
# 
# # Plot
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
# print("Top 6 indices with the highest average absolute values:", top_indices)
# # we see that when:
# # running for gold dict: [ 194  954 1120 1246 1348 1487]
# # running for gpt: [ 194  954 1120 1246 1487 1348]
# # different embeddings indices seem to be most informative
# =============================================================================


########################################


# STEP 3.2 (optional) Plot one (informative) dimension of the embeddings 


# Print the top 6 indices with the highest average absolute values: [ 194  954 1120 1246 1348 1487] all minus 1! 
#plot_embeddings_1D_to_html(embeddings_gd, embeddings_gpt, gold_dict_bsb_filtered, gpt_clean_bsb_filtered, 193)


########################################


# STEP 4. Computing similarity: 

compare_results = compare_embeddings(embeddings_gd, embeddings_gpt)  
plot_distribution_metrics(compare_results)


########################################


# STEP 5. Plotting similarity distribution - with metadata 

def plot_distribution_with_spread_details_fixed(compare_results, embeddings_dict1, embeddings_dict2, labels_dict1, labels_dict2, distance_type):
    
    sample_distances = [(key, result[distance_type]) for key, result in compare_results.items()]
    sample_ids = [key for key, _ in sample_distances]

    color_map = {
        'animal': '#C67D7B',  # Pinkish
        'water': '#8CC8CF',   # Bluish
        'plant': '#C0D184',   # Greenish
        'soil': '#CBBF82',    # Brownish
        'other': '#CCCCCC'    # Gray
    }
    colors = [color_map.get(labels_dict1[key].split(' - ')[0], '#CCCCCC') for key in sample_ids]
    hover_texts = [f"{key}:<br>Gold: {labels_dict1[key]}<br>GPT: {labels_dict2[key]}" for key in sample_ids]

    distances = [dist for _, dist in sample_distances]
    bin_size = np.ptp(distances) / 30  # bin size
    bins = np.arange(min(distances), max(distances) + bin_size, bin_size)
    hist = np.histogram(distances, bins=bins)
    bin_edges = hist[1]
    bin_counts = hist[0]
    
    # positions for scatter points
    bin_assignment = [np.digitize(dist, bin_edges) - 1 for _, dist in sample_distances]
    y_positions = {bin_idx: [] for bin_idx in range(len(bin_counts))}
    
    for idx, bin_idx in enumerate(bin_assignment):
        if bin_idx == len(bin_counts):  # handle edge case where value falls on the last bin edge
            bin_idx -= 1
        y_positions[bin_idx].append(sample_distances[idx][0])  # store sample ID
    
    scatter_x = []
    scatter_y = []
    scatter_colors = []
    scatter_texts = []
    
    for bin_idx, ids in y_positions.items():
        heights = np.linspace(1, bin_counts[bin_idx], num=len(ids))  # evenly spread out within the bin
        for i, id in enumerate(ids):
            scatter_x.append(compare_results[id][distance_type])
            scatter_y.append(heights[i])
            scatter_colors.append(colors[sample_ids.index(id)])
            scatter_texts.append(hover_texts[sample_ids.index(id)])
    
    fig = go.Figure()
    hist_trace = go.Histogram(x=distances, xbins=dict(start=bins[0], end=bins[-1], size=bin_size), opacity=0.75, name=f'{distance_type.capitalize()} Distance')
    fig.add_trace(hist_trace)
    
    # create scatter plot on top of histogram
    fig.add_trace(go.Scatter(x=scatter_x, y=scatter_y, mode='markers', text=scatter_texts, hoverinfo='text', marker=dict(color=scatter_colors, size=5)))
    # text annotations on the histogram bars to show the count of samples within each bin
    for i, bin_count in enumerate(bin_counts):
        if bin_count > 0:
            # calculate the position for the annotation to be in the middle of the bin
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            fig.add_annotation(
                x=bin_center, y=bin_count + 1,  # small offset above the bar for clarity
                text=str(bin_count),
                showarrow=False,
                font=dict(color="black", size=12)
            )
    
    fig.update_layout(title=f"{distance_type.capitalize()} Cosine similarity sistribution with sample metadata",
                      xaxis_title=f"{distance_type.capitalize()} cosine similarity",
                      yaxis_title='Samples (spread) within bin',
                      hovermode='closest',
                      showlegend=False)
    fig.show()
    fig.write_html(f"/Users/dgaio/MicrobeAtlasProject/20240423_biome_and_subbiomes_similarity_{distance_type}.html")
    

plot_distribution_with_spread_details_fixed(compare_results, embeddings_gd, embeddings_gpt, gold_dict_bsb_filtered, gpt_clean_bsb_filtered, 'cosine')



########################################


# STEP 5.1 Plotting cosine similarity vs background 

actual_similarities = [result['cosine'] for result in compare_results.values()]
num_comparisons = len(actual_similarities)  

background_similarities = create_shuffled_background_distribution(embeddings_gd, embeddings_gpt, num_comparisons=num_comparisons)

plot_comparison_distribution(actual_similarities, background_similarities)



########################################


# STEP 5.1 Plotting cosine similarity - Heatmap 

n_samples_per_category = 10  # Adjust this number as needed

# Sample keys from the gold dictionary
keys_gd_sampled = sample_by_category(gold_dict_bsb_filtered, n_samples_per_category)
# Assume the test keys align exactly with the sampled ground truth keys
keys_gpt_sampled = [key for key in keys_gd_sampled]  # Ensuring alignment

# Construct the matrices for the sampled keys
matrix_gd = np.array([embeddings_gd[key] for key in keys_gd_sampled])
matrix_gpt = np.array([embeddings_gpt[key] for key in keys_gpt_sampled])

# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(matrix_gd, matrix_gpt)

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.set(font_scale=0.7)  # Slightly larger font scale for readability
ax = sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm',
                 xticklabels=[gpt_clean_bsb_filtered[key] for key in keys_gpt_sampled], 
                 yticklabels=[gold_dict_bsb_filtered[key] for key in keys_gd_sampled])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

plt.title('Cosine Similarity Heatmap Grouped by Category')
plt.xlabel('Test Samples')
plt.ylabel('Ground Truth Samples')
plt.show()


