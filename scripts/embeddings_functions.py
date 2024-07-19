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
from scipy.stats import ttest_rel, mannwhitneyu
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests


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
    std_dev = np.std(similarities)
    percentiles = np.percentile(similarities, [25, 50, 75])
    
    print(f"Average cosine similarity: {avg_sim:.4f}")
    print(f"Median cosine similarity: {median_sim:.4f}")
    print(f"Standard deviation of cosine similarity: {std_dev:.4f}")
    print(f"Percentiles: {percentiles}")
    return avg_sim, median_sim, std_dev, percentiles



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
    #plt.show()




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



def test_similarity_separation(actual_similarities, background_similarities):
    """Performs a statistical test to see if actual and background similarities are significantly different and returns the p-value."""
    stat, p_value = mannwhitneyu(actual_similarities, background_similarities)
    print(f"Actual vs background similarities: Mann-Whitney U test: U={stat}, p-value={p_value}")
    return stat, p_value
 


def plot_actual_vs_background(actual_similarities, background_similarities, title, avg_sim, median_sim, std_dev, MWU_stat, MWU_p_value):
    """Plots a box plot comparing actual and background cosine similarities and includes p-value."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([actual_similarities, background_similarities], notch=True, patch_artist=True, labels=['Actual', 'Background'])
    ax.set_title(title)
    ax.set_ylabel('Cosine Similarity')
    ax.text(0.95, 0.95, f'''
            avg: {avg_sim:.2f} 
            sd: {std_dev:.2f} 
            med: {median_sim:.2f} 
            MannWhitney U test\nU: {MWU_stat:.4f}
            MannWhitney U test\np-value: {MWU_p_value:.4f}
            ''', 
            horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=10)
    ax.grid(True)
    #plt.show()
    return fig



def compare_based_on_overlap(similarities_dict1, similarities_dict2, threshold=0.7):
    keys1 = set(similarities_dict1.keys())
    keys2 = set(similarities_dict2.keys())
    common_keys = keys1 & keys2
    total_keys = keys1 | keys2
    overlap_percentage = len(common_keys) / len(total_keys)
    print('Percentage of overlapping samples: ', overlap_percentage*100)
    
    sorted_common_keys = sorted(common_keys)
    similarities1 = [similarities_dict1[key]['cosine'] for key in sorted_common_keys]
    similarities2 = [similarities_dict2[key]['cosine'] for key in sorted_common_keys]

    if overlap_percentage >= threshold:
        stat, p_value = ttest_rel(similarities1, similarities2)
        test_type = 'ttest_rel'
    else:
        stat, p_value = ttest_ind(similarities1, similarities2)
        test_type = 'ttest_ind'

    num_tests=len(sorted_common_keys)
    p_adjusted = min(p_value * num_tests, 1.0)  # ensures p-value does not exceed 1
    
    print(f"{test_type} t-test result: t={stat}, p={p_value}, p-adj={p_adjusted}")
    return overlap_percentage*100, stat, p_value, p_adjusted, test_type



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


def save_figures_to_pdf(figures, base_filename, directory):
    """Saves a list of figure objects to a PDF file in the specified directory."""
    import matplotlib.backends.backend_pdf
    pdf_path = os.path.join(directory, f"{base_filename}.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()
    print(f"Saved to {pdf_path}")

        
        