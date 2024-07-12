#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""

import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from features_process import filter_common_keys

from embeddings_functions import load_embeddings
from embeddings_functions import compare_embeddings
from embeddings_functions import print_statistics
from embeddings_functions import plot_distribution_metrics
from embeddings_functions import plot_comparison_distribution
from embeddings_functions import create_shuffled_background_distribution
from embeddings_functions import sample_by_category
from embeddings_functions import plot_heatmap
from embeddings_functions import save_figures_to_pdf

import os
import numpy as np
import random
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel


# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'

###
# Gold dict path 
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')  
embeddings_gd = load_embeddings(gold_dict_json_path)
###

########
# expected different
my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']


# expected similar
# my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
#             'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt']


gpt_json_files=[]
for i in my_files:
   j=i.replace('.txt', '_sbembeddings.json')
   gpt_json_files.append(j)
########



# Fetch embeddings from each gpt json file and compare to ground truth embeddings:
for gpt_file in gpt_json_files:
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    embeddings_gpt = load_embeddings(gpt_json_path)
    
    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    print("Filtered gold_dict_sb:", len(filtered_gd))
    print("Filtered gpt_clean_sb:", len(filtered_gpt))

    ########################################
    # Compare embeddings
    compare_results = compare_embeddings(filtered_gd, filtered_gpt)

    ########################################
    # Calculate and print statistics
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim = print_statistics(actual_similarities)

    ########################################
    # Plot distribution metrics
    plot_distribution_metrics(compare_results)

    ########################################
    # Plot similarity vs background
    background_similarities = create_shuffled_background_distribution(filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities))
    comparison_fig = plot_comparison_distribution(actual_similarities, background_similarities, avg_sim, median_sim)

    ########################################
    # Extract sub_biome_texts directly from embeddings dictionaries
    gold_labels = {key: embeddings_gd[key]['sub-biome'] for key in embeddings_gd}
    gpt_labels = {key: embeddings_gpt[key]['sub-biome'] for key in embeddings_gpt}
    
    ### # filtering to keep only common keys in both dictionaries, and 10 per biome
    gold_biomes = {key: embeddings_gd[key]['biome'] for key in embeddings_gd}
    common_keys = list(gold_labels.keys() & gpt_labels.keys())
    sampled_keys = sample_by_category(common_keys, gold_biomes, 10)
    ###
    
    
    # Prepare data matrices for the heatmap
    matrix_gd = np.array([embeddings_gd[key]['embedding'] for key in sampled_keys])
    matrix_gpt = np.array([embeddings_gpt[key]['embedding'] for key in sampled_keys])

    # Extract the labels for the sampled keys
    gold_labels_sampled = {key: gold_labels[key] for key in sampled_keys}
    gpt_labels_sampled = {key: gpt_labels[key] for key in sampled_keys}

    # Generate heatmap
    heatmap_fig = plot_heatmap(matrix_gd, matrix_gpt, gpt_labels_sampled, gold_labels_sampled, sampled_keys, sampled_keys)


    ########################################
    # Save both figures to a PDF
    gpt_base_file = gpt_file.replace('_sbembeddings.json', '')
    save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, embeddings_dir)
    
    
    
    
    
    
    
    
    
    






# File paths
my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp1.0_presp1.5_rs22_API132_normal_dt202406051450.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu

sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from embeddings_functions import load_embeddings, compare_embeddings
from features_process import filter_common_keys
from embeddings_functions import print_statistics, plot_distribution_metrics, plot_comparison_distribution
from embeddings_functions import create_shuffled_background_distribution, sample_by_category
from embeddings_functions import plot_heatmap, save_figures_to_pdf

# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')
embeddings_gd = load_embeddings(gold_dict_json_path)


def find_distinguishing_features(files):
    """
    Determine the distinguishing features between filenames.
    Collect all tokens and identify those that are unique to some but not all filenames.
    """
    all_tokens = []
    file_tokens = []

    for file in files:
        tokens = os.path.basename(file).split('_')[:-2]  # to exclude date and time
        file_tokens.append(set(tokens))
        all_tokens.extend(tokens)

    token_count = {}
    for token in set(all_tokens):
        token_count[token] = sum(1 for tokens in file_tokens if token in tokens)

    # find tokens that are unique to some files but not common to all
    num_files = len(files)
    distinguishing_tokens = {token for token, count in token_count.items() if count != num_files}

    return distinguishing_tokens


def extract_labels_from_filename(filename, distinguishing_tokens):
    """
    Extract distinguishing labels from the filename based on the distinguishing tokens.
    """
    tokens = os.path.basename(filename).split('_')[:-2]  # to exclude date and time
    labels = [token for token in tokens if token in distinguishing_tokens]
    return ", ".join(labels)



# Use the function to determine distinguishing features
distinguishing_tokens = find_distinguishing_features(my_files)

# Extract labels for plotting based on distinguishing features
labels = [extract_labels_from_filename(f, distinguishing_tokens) for f in my_files]

gpt_json_files = [os.path.join(embeddings_dir, f.replace('.txt', '_sbembeddings.json')) for f in my_files]

def compare_embeddings_filtered(gold_embeddings, gpt_embeddings, threshold=0.75):
    compare_results = {}
    for key in gold_embeddings:
        if key in gpt_embeddings:
            gold_vector = gold_embeddings[key]['embedding']
            gpt_vector = gpt_embeddings[key]['embedding']
            cos_sim = 1 - cosine(gold_vector, gpt_vector)
            if cos_sim >= threshold:
                compare_results[key] = {'cosine': cos_sim}
    return compare_results

def print_statistics(similarities):
    if similarities:
        average_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        print(f"Average similarity: {average_similarity:.3f}")
        print(f"Median similarity: {median_similarity:.3f}")
        return average_similarity, median_similarity
    else:
        print("No similarities above the threshold.")
        return 0, 0

all_similarities = []  # List to hold all similarity lists for statistical comparison

# Main execution block
for gpt_file in gpt_json_files:
    print("Processing file:", gpt_file)
    embeddings_gpt = load_embeddings(gpt_file)
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    compare_results = compare_embeddings_filtered(filtered_gd, filtered_gpt)
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim = print_statistics(actual_similarities)
    all_similarities.append(actual_similarities)

# Plotting and Statistical Test
if len(all_similarities) > 1:
    # Perform statistical test
    stat, p_value = mannwhitneyu(all_similarities[0], all_similarities[1], alternative='two-sided')
    print(f"Mann-Whitney U test statistic: {stat}, P-value: {p_value}")

    # Visualization
    fig, ax = plt.subplots()
    bp = ax.boxplot(all_similarities, labels=labels)
    plt.ylabel('Cosine Similarity')
    plt.title('Comparison of Cosine Similarities Between Two GPT Runs')
    plt.xticks(rotation=45)  # Rotate labels for better readability

    # Annotating the sample size
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]  # top of median line
        ax.annotate(f'n={len(all_similarities[i])}', xy=(x, y), xytext=(0,5), 
                    textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()

else:
    print("Not enough data to perform statistical comparison.")



# test normality in continuous data: Shapiro-Wilk Test for Normality:

    
# 2 files; independent samples; skewed data; continuous data --> Mann-Whitney
# > 2 files, independent samples; skewed data; continuous data --> Kruskal-Wallis One-Way analysis of variance (ANOVA) 

# 2 files; dependent; skewed/normal; continuous data --> Wilcoxon Signed-Rank Test (checks whether the differences between paired observations are symmetrically distributed around zero)
# > 2 files, dependent: Friedman test


import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import friedmanchisquare, shapiro
import os

def filter_common_keys_across_groups(groups):
    # Using set intersection to find common keys
    common_keys = set.intersection(*(set(group.keys()) for group in groups))
    print(f"Number of common keys: {len(common_keys)}")  # Debug: Check how many keys are common

    # Filter each group to retain only the common keys
    filtered_groups = [{key: group[key] for key in common_keys} for group in groups]
    
    # Debug: Check the size of each group after filtering
    for i, fg in enumerate(filtered_groups, 1):
        print(f"Size of Group {i}: {len(fg)}")
    
    return filtered_groups

def compare_embeddings_filtered(gold_embeddings, gpt_embeddings, threshold=0.75):
    compare_results = {}
    for key in gold_embeddings:
        if key in gpt_embeddings:
            gold_vector = gold_embeddings[key]['embedding']
            gpt_vector = gpt_embeddings[key]['embedding']
            cos_sim = 1 - cosine(gold_vector, gpt_vector)
            compare_results[key] = {'cosine': cos_sim}  # Always return the cosine similarity
    return compare_results


# Assuming embeddings_groups is a list of dictionaries where each dictionary is a group of embeddings
embeddings_groups = [load_embeddings(gpt_file) for gpt_file in gpt_json_files]

# Filter to ensure all groups have the same keys
filtered_groups = filter_common_keys_across_groups(embeddings_groups)


all_similarities = []
for group in filtered_groups:
    filtered_gd = {key: embeddings_gd[key] for key in group.keys() if key in embeddings_gd}
    compare_results = compare_embeddings_filtered(filtered_gd, group)
    similarities = [result['cosine'] for result in compare_results.values()]
    print(len(similarities))
    all_similarities.append(similarities)
    

# Check all groups have the same number of entries
if all(len(sim) == len(all_similarities[0]) for sim in all_similarities):
    # Proceed with Friedman test
    stat, p_value = friedmanchisquare(*all_similarities)
    print(f"Friedman test statistic: {stat}, P-value: {p_value}")
else:
    print("Error: Not all groups have the same number of entries.")




