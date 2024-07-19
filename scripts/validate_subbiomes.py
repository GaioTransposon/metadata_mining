#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""

import pandas as pd
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



########

# -----------------------------
# Files processing
# ----------------------------- 
 
features = find_distinguishing_features(my_files)
file_label_map = {file: extract_labels_from_filename(file, features) for file in my_files}
file_label_map = edit_features(file_label_map)

print("\nFile and its label name:\n")
for file, label in file_label_map.items():
    print(f"{os.path.basename(file)} - {label}\n")




results={}
results_sub_biome = []
results_list = []

# Fetch embeddings from each gpt json file and compare to ground truth embeddings:
for gpt_file in my_files:    
    
    gpt_file = re.sub(r'\.txt|\.csv', '_sbembeddings.json', gpt_file)
    gpt_json_file_path = os.path.join(embeddings_dir, gpt_file) 
    embeddings_gpt = load_embeddings(gpt_json_file_path)
    
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
    MWU_stat, MWU_p_value = test_similarity_separation(actual_similarities, background_similarities)
    title = f"Comparison of Actual vs Background Cosine Similarity for\n{gpt_file}"
    comparison_fig = plot_actual_vs_background(actual_similarities, background_similarities, title, 
                                               avg_sim, median_sim, std_dev,
                                               MWU_stat, MWU_p_value)
    
    ########################################
    # Gather info: 
    #avg_sim, median_sim, std_dev, percentiles, MWU_stat, MWU_p_value, filename, label
    results_sub_biome = {
        #'Label': file_label_map[os.path.join(embeddings_dir, gpt_file.replace('_sbembeddings.json', '.txt'))],
        'Average Similarity': avg_sim,
        'Median Similarity': median_sim,
        'Standard Deviation': std_dev,
        '95th Percentile': percentiles,  # Assuming you want the 95th percentile
        'MWU Statistic': MWU_stat,
        'MWU P-value': MWU_p_value,
        'Filename': gpt_file,
    }
    
    # Append the dictionary to the results list
    results_list.append(results_sub_biome)
    
    ########################################
    # Plotting: 
        
    # Extract sub_biome_texts directly from embeddings dictionaries
    gold_labels = {key: embeddings_gd[key]['sub-biome'] for key in embeddings_gd}
    gpt_labels = {key: embeddings_gpt[key]['sub-biome'] for key in embeddings_gpt}
    
    ### filtering to keep only common keys in both dictionaries, and 10 per biome
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

    
# concatenate data 
results_df = pd.DataFrame(results_list)
print(results_df)
# colnames are: 	Average Similarity	Median Similarity	Standard Deviation	95th Percentile	MWU Statistic Filename	




# -----------------------------
# Stats
# ----------------------------- 

results_data = []

# Stats pairwise comparisons with dynamic test selection based on overlap (if more than 70% samples in common, dependent test)
for file1, file2 in combinations(results.keys(), 2):
    print(f"\n\nComparing file:\n\n{file1}\nwith file:\n{file2}\n")
    overlap_percentage, stat, p_value, p_adjusted, test_type = compare_based_on_overlap(results[file1], results[file2])

    result_dict = {
        # label1 and labels2
        'Statistic': stat,
        'P-value': p_value,
        'Adjusted P-value': p_adjusted,
        'Test Type': test_type,
        'Filename1': file1,
        'Filename2': file2
    }

    results_data.append(result_dict)
    
results_df_stats = pd.DataFrame(results_data)
results_df_stats['validation'] = 'sub-biome'

print(results_df_stats)
# colnames are: 	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2












