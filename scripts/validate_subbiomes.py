#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""

import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from embeddings_functions import load_embeddings
from embeddings_functions import compare_embeddings
from features_process import filter_common_keys
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

def compare_embeddings_filtered(gold_embeddings, gpt_embeddings, threshold=0.75):
    compare_results = {}
    for key in gold_embeddings:
        if key in gpt_embeddings:
            # Extract embeddings
            gold_vector = gold_embeddings[key]['embedding']
            gpt_vector = gpt_embeddings[key]['embedding']
            
            # Calculate cosine similarity (Note: scipy's cosine returns the cosine distance, not similarity)
            cos_sim = 1 - cosine(gold_vector, gpt_vector)
            
            # Apply threshold filter
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
    
    
    
    
    
    
    
    

# Main execution block
for gpt_file in gpt_json_files:
    
    print(gpt_json_path)
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    embeddings_gpt = load_embeddings(gpt_json_path)
    
    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    
    # Compare embeddings using the new filtered function
    compare_results = compare_embeddings_filtered(filtered_gd, filtered_gpt)
    
    # Calculate and print statistics based on the new comparison results
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim = print_statistics(actual_similarities)
    
    # Visualize and analyze further as needed


















