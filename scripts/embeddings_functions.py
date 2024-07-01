#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:28:44 2024

@author: dgaio
"""


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_pdf import PdfPages
import random


# Now adapted to load embeddings with this format: {sample_id: {'embedding': [values], 'sub_biome_text': text}}
def load_embeddings(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Create a dictionary to store embeddings, sub_biome_text, and biome (if available)
    embeddings_dict = {}
    for k, v in data.items():
        embeddings = np.array(v['embedding'], dtype=np.float32)
        sub_biome_text = v['sub-biome']
        embeddings_dict[k] = {'embedding': embeddings, 'sub-biome': sub_biome_text}
        
        # Include biome if it exists in the data
        if 'biome' in v:
            embeddings_dict[k]['biome'] = v['biome']

    return embeddings_dict


def compare_embeddings(embeddings_dict1, embeddings_dict2):
    comparison_results = {}
    for sample_id, data1 in embeddings_dict1.items():
        if sample_id in embeddings_dict2:
            # Extract embedding arrays from each dictionary
            embedding1 = data1['embedding']
            embedding2 = embeddings_dict2[sample_id]['embedding']
            
            # Compute distances and similarity
            euclidean_dist = distance.euclidean(embedding1, embedding2)
            cosine_sim = 1 - distance.cosine(embedding1, embedding2)  # cosine similarity is 1 - cosine distance
            manhattan_dist = distance.cityblock(embedding1, embedding2)
            
            # Store results
            comparison_results[sample_id] = {
                'euclidean': euclidean_dist,
                'cosine': cosine_sim, 
                'manhattan': manhattan_dist
            }
    return comparison_results


def print_statistics(similarities):
    avg_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    print(f"Average cosine similarity: {avg_sim:.4f}")
    print(f"Median cosine similarity: {median_sim:.4f}")
    return avg_sim, median_sim


def plot_distribution_metrics(compare_results):
    euclidean_distances = [result['euclidean'] for result in compare_results.values()]
    cosine_similarities = [result['cosine'] for result in compare_results.values()]
    manhattan_distances = [result['manhattan'] for result in compare_results.values()]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.histplot(euclidean_distances, bins=30, kde=True, ax=axs[0], color='blue')
    axs[0].set_title('Euclidean distance distribution')
    sns.histplot(cosine_similarities, bins=30, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Cosine similarity distribution')
    sns.histplot(manhattan_distances, bins=30, kde=True, ax=axs[2], color='red')
    axs[2].set_title('Manhattan distance distribution')
    plt.tight_layout()
    plt.show()


def plot_comparison_distribution(actual_similarities, background_similarities, avg_sim, median_sim):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(actual_similarities, bins=30, kde=True, color='green', label='Actual Cosine Similarities', ax=ax, stat='density')
    sns.histplot(background_similarities, bins=30, kde=True, color='blue', label='Background Cosine Similarities', ax=ax, alpha=0.5, stat='density')
    ax.set_title('Actual vs Background Cosine Similarities')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Probability Density')
    ax.legend()
    plt.text(0.95, 0.95, f'Avg: {avg_sim:.4f}\nMed: {median_sim:.4f}', verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=12)
    plt.tight_layout()
    return fig  


def create_shuffled_background_distribution(embeddings_gd, embeddings_gpt, num_comparisons=None):
    random_cosine_similarities = []
    gd_keys = list(embeddings_gd.keys())
    gpt_keys = list(embeddings_gpt.keys())
    
    # Shuffle the keys for randomness in comparison
    random.shuffle(gpt_keys)
    
    # If num_comparisons is not specified, compare the full length of the smaller set
    if num_comparisons is None:
        num_comparisons = min(len(gd_keys), len(gpt_keys))
    else:
        num_comparisons = min(num_comparisons, len(gd_keys), len(gpt_keys))
    
    for i in range(num_comparisons):
        # Extract embeddings from each dictionary by specifying the 'embedding' key
        gd_embedding = embeddings_gd[gd_keys[i]]['embedding']
        gpt_embedding = embeddings_gpt[gpt_keys[i]]['embedding']
        
        # Calculate cosine similarity
        cosine_sim = 1 - distance.cosine(gd_embedding, gpt_embedding)
        random_cosine_similarities.append(cosine_sim)
    
    return random_cosine_similarities



def sample_by_category(common_keys, gold_biomes, n):
    random.seed(42)  # For reproducibility
    
    # Group common keys by biome
    category_groups = {}
    for key in common_keys:
        category = gold_biomes[key]  # Use the 'biome' value as the category
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(key)

    sampled_keys = []
    for category, keys in category_groups.items():
        if len(keys) > n:
            sampled_keys.extend(random.sample(keys, n))
        else:
            sampled_keys.extend(keys)  # If less than n, take all
    return sampled_keys




def plot_heatmap(matrix_gd, matrix_gpt, gpt_labels, gold_labels, keys_gpt_sampled, keys_gd_sampled):
    similarity_matrix = cosine_similarity(matrix_gd, matrix_gpt)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.set(font_scale=0.5)
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', ax=ax,
                          xticklabels=[gpt_labels[key] for key in keys_gpt_sampled],
                          yticklabels=[gold_labels[key] for key in keys_gd_sampled])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    plt.title('Cosine Similarity Heatmap Grouped by Category')
    plt.xlabel('Test Samples')
    plt.ylabel('Ground Truth Samples')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.20, left=0.15, right=1.00)
    #plt.show()
    return fig  

# Function to save all figures to a single PDF
def save_figures_to_pdf(figures, file_name, directory):
    file_path = os.path.join(directory, file_name + '.pdf')
    with PdfPages(file_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            #plt.close(fig)  
        print(f"All plots saved to {file_path}")
        
        
        