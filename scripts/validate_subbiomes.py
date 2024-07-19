#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""


import os
import numpy as np
import re
from itertools import combinations
import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')

from features_process import filter_common_keys
from embeddings_functions import (load_embeddings, compare_embeddings, print_statistics,
                                  create_shuffled_background_distribution,
                                  sample_by_category, plot_heatmap, save_figures_to_pdf,
                                  plot_actual_vs_background,
                                  test_similarity_separation, compare_based_on_overlap)

import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode


# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'

###
# Gold dict path 
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')  
embeddings_gd = load_embeddings(gold_dict_json_path)
###

########
# # expected different
# my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
#             'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']

# expected similar
# my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
#             'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt']

gpt_json_files = [re.sub(r'\.txt|\.csv', '_sbembeddings.json', f) for f in my_files]

########

results = {}

# Fetch embeddings from each gpt json file and compare to ground truth embeddings:
for gpt_file in gpt_json_files:
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    embeddings_gpt = load_embeddings(gpt_json_path)
    
    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    print("\nSample size after filtering:", len(filtered_gpt))

    ########################################
    # Compare embeddings
    compare_results = compare_embeddings(filtered_gd, filtered_gpt)

    ########################################
    # Calculate and print statistics
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim, std_dev, percentiles = print_statistics(actual_similarities)
    
    results[gpt_file] = compare_results

    
    ########################################
    # Similarity vs background
    background_similarities = create_shuffled_background_distribution(filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities))
    p_value = test_similarity_separation(actual_similarities, background_similarities)
    title = f"Comparison of Actual vs Background Cosine Similarity for\n{gpt_file}"
    comparison_fig = plot_actual_vs_background(actual_similarities, background_similarities, title, 
                                               avg_sim, median_sim, std_dev,
                                               p_value)
    
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
    gold_labels_sampled = {key: gold_labels[key] for key in sampled_keys}
    gpt_labels_sampled = {key: gpt_labels[key] for key in sampled_keys}

    # Generate heatmap
    heatmap_fig = plot_heatmap(matrix_gd, matrix_gpt, gpt_labels_sampled, gold_labels_sampled, sampled_keys, sampled_keys)


    ########################################
    # Save both to a PDF
    gpt_base_file = gpt_file.replace('_sbembeddings.json', '')
    save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, embeddings_dir)

    

# Stats pairwise comparisons with dynamic test selection based on overlap (if more than 70% samples in common, dependent test)
for file1, file2 in combinations(results.keys(), 2):
    print(f"\n\nComparing file:\n\n{file1}\nwith file:\n{file2}\n")
    compare_based_on_overlap(results[file1], results[file2])



