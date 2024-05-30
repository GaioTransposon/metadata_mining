#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""


from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import csv
import os
from matplotlib.backends.backend_pdf import PdfPages



def load_embeddings(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    embeddings_dict = {k: np.array(v, dtype=np.float32) for k, v in data['embeddings'].items()}
    failed_samples = data['failed_samples']
    return embeddings_dict, failed_samples


def load_labels_gpt(csv_file_path):
    samples = {}
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header 
        for row in reader:
            # skip files taht dont have at leats 5 columns
            if len(row) < 5:
                continue  
            sample_id = row[0]
            combined_text = f"{row[1]} - {row[4]}"
            samples[sample_id] = combined_text
    return samples

def load_labels_gold_dict(gold_dict_path):
    gold_dict_bsb = {}
    with open(gold_dict_path, 'rb') as file:
        data = pickle.load(file)
        for key, values in data.items():
            if len(values) >= 3:
                combined_text = f"{values[1]} - {values[2]}"
                #print(key)  
                #print(combined_text)  
                gold_dict_bsb[key] = combined_text
    return gold_dict_bsb


def filter_common_keys(embeddings_dict1, embeddings_dict2):
    """ Filter embeddings to only include common keys. """
    common_keys = set(embeddings_dict1.keys()) & set(embeddings_dict2.keys())
    filtered_dict1 = {k: embeddings_dict1[k] for k in common_keys}
    filtered_dict2 = {k: embeddings_dict2[k] for k in common_keys}
    return filtered_dict1, filtered_dict2


def compare_embeddings(embeddings_dict1, embeddings_dict2):
    comparison_results = {}
    for sample_id, embedding1 in embeddings_dict1.items():
        if sample_id in embeddings_dict2:
            embedding2 = embeddings_dict2[sample_id]
            euclidean_dist = distance.euclidean(embedding1, embedding2)
            cosine_sim = 1 - distance.cosine(embedding1, embedding2)  # cosine similarity is 1 - cosine distance
            manhattan_dist = distance.cityblock(embedding1, embedding2)
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
    
    random.shuffle(gpt_keys)
    num_comparisons = min(len(gd_keys), len(gpt_keys))
    
    for i in range(num_comparisons):
        gd_embedding = embeddings_gd[gd_keys[i]]
        gpt_embedding = embeddings_gpt[gpt_keys[i]]
        cosine_sim = 1 - distance.cosine(gd_embedding, gpt_embedding)
        random_cosine_similarities.append(cosine_sim)
    
    return random_cosine_similarities



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
    plt.subplots_adjust(top=0.95, bottom=0.14, left=0.15, right=1.00)
    plt.show()
    return fig  



# Function to save all figures to a single PDF
def save_figures_to_pdf(figures, file_name, directory):
    file_path = os.path.join(directory, file_name.replace('.json', '.pdf'))
    with PdfPages(file_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)  
        print(f"All plots saved to {file_path}")
        
        
########################################

# Fetch embeddings: 

# File to test: 
gpt_file = 'gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs33_batchgTBuqJNA7w30eIjHax535YMQ_dt20240524_1537'


gpt_json_path = gpt_file + '_bsbembeddings.json'
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'
gpt_json_path = os.path.join(work_dir, gpt_json_path)

gold_dict_json_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings/gold_dict_bsbembeddings.json'  

embeddings_gpt, failed_samples_gpt = load_embeddings(gpt_json_path)
embeddings_gd, failed_samples_gd = load_embeddings(gold_dict_json_path)

# Filter embeddings to include only common keys
filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
print("Filtered gold_dict_bsb:", len(filtered_gd))
print("Filtered gpt_clean_bsb:", len(filtered_gpt))


########################################


# Compute and plot similarities
compare_results = compare_embeddings(filtered_gd, filtered_gpt)
#plot_distribution_metrics(compare_results)

########################################


# Plotting cosine similarity vs background 

actual_similarities = [result['cosine'] for result in compare_results.values()]
background_similarities = create_shuffled_background_distribution(embeddings_gd, embeddings_gpt, num_comparisons=len(actual_similarities))

# Calculate statistics
avg_sim, median_sim = print_statistics(actual_similarities)


# Plot similarity vs background
comparison_fig = plot_comparison_distribution(actual_similarities, background_similarities, avg_sim, median_sim)

########################################


# Heatmap 

gpt_file = gpt_file + '.csv'
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'
gpt_path = os.path.join(work_dir, gpt_file)
gold_path = '/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl'


gpt_labels = load_labels_gpt(gpt_path)
gold_labels = load_labels_gold_dict(gold_path)

gold_labels, gpt_labels = filter_common_keys(gold_labels, gpt_labels)

n_samples_per_category = 10  # Adjust this number as needed

# use these labels in sampling function
keys_gd_sampled = sample_by_category(gold_labels, n_samples_per_category)
keys_gpt_sampled = [key for key in keys_gd_sampled]  # ensuring alignment

matrix_gd = np.array([embeddings_gd[key] for key in keys_gd_sampled])
matrix_gpt = np.array([embeddings_gpt[key] for key in keys_gpt_sampled])


# Plot heatmap
heatmap_fig = plot_heatmap(matrix_gd, matrix_gpt, gpt_labels, gold_labels, keys_gpt_sampled, keys_gd_sampled)


########################################

# Save both figures to a PDF
save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_json_path, work_dir)







