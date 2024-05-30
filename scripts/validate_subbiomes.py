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

# Fetch embeddings: 

# Define paths to the JSON files containing embeddings
gpt_json_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings/gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs33_batchgTBuqJNA7w30eIjHax535YMQ_dt20240524_1537_bsbembeddings.json'  
gold_dict_json_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings/gold_dict_bsbembeddings.json'  

embeddings_gpt, failed_samples_gpt = load_embeddings(gpt_json_path)
embeddings_gd, failed_samples_gd = load_embeddings(gold_dict_json_path)

# Filter embeddings to include only common keys
filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
print("Filtered gold_dict_bsb:", len(filtered_gd))
print("Filtered gpt_clean_bsb:", len(filtered_gpt))


########################################


# Compute similarities
compare_results = compare_embeddings(filtered_gd, filtered_gpt)
plot_distribution_metrics(compare_results)


########################################


# Plotting cosine similarity vs background 

actual_similarities = [result['cosine'] for result in compare_results.values()]
num_comparisons = len(actual_similarities)  

background_similarities = create_shuffled_background_distribution(embeddings_gd, embeddings_gpt, num_comparisons=num_comparisons)

plot_comparison_distribution(actual_similarities, background_similarities)


########################################


# Heatmap 

# labels from original files: 
gpt_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs33_batchgTBuqJNA7w30eIjHax535YMQ_dt20240524_1537.csv'
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

similarity_matrix = cosine_similarity(matrix_gd, matrix_gpt)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.set(font_scale=0.7)
ax = sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm',
                 xticklabels=[gpt_labels[key] for key in keys_gpt_sampled],
                 yticklabels=[gold_labels[key] for key in keys_gd_sampled])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

plt.title('Cosine Similarity Heatmap Grouped by Category')
plt.xlabel('Test Samples')
plt.ylabel('Ground Truth Samples')
plt.show()


########################################


