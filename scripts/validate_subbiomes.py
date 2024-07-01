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
from features_process import load_bsb_gpt
from features_process import load_bsb_gold_dict
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


# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'
all_files = os.listdir(embeddings_dir)

# Filter to get only files that start with 'gpt' and end with '.json'
gpt_json_files = [f for f in all_files if f.startswith('gpt') and f.endswith('.json')]


# Gold dict path 
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_bsbembeddings.json')  
embeddings_gd = load_embeddings(gold_dict_json_path)

# Parent directory to access CSV files
parent_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'


# Fetch embeddings from each gpt json file and compare to ground truth embeddings:
for gpt_file in gpt_json_files:
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    embeddings_gpt = load_embeddings(gpt_json_path)
    
    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    print("Filtered gold_dict_bsb:", len(filtered_gd))
    print("Filtered gpt_clean_bsb:", len(filtered_gpt))

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
    # Setup for heatmap (cause we need to retrieve labels)
    # Correctly adjust filename to match CSV naming convention
    gpt_base_file = gpt_file.replace('_bsbembeddings.json', '')
    gpt_csv_path = os.path.join(parent_dir, gpt_base_file + '.csv')
    gpt_txt_path = os.path.join(parent_dir, gpt_base_file + '.txt')
    
    
    if os.path.exists(gpt_csv_path):
        gpt_path = gpt_csv_path
    elif os.path.exists(gpt_txt_path):
        gpt_path = gpt_txt_path
    else:
        continue  # Skip if neither file exists

    gpt_labels = load_bsb_gpt(gpt_path)
    
    gold_path = '/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl'
    gold_labels = load_bsb_gold_dict(gold_path)
    gold_labels, gpt_labels = filter_common_keys(gold_labels, gpt_labels)

    # Sample keys by category and prepare data for heatmap
    keys_gd_sampled = sample_by_category(gold_labels, 10)
    keys_gpt_sampled = [key for key in keys_gd_sampled]  # ensuring alignment
    matrix_gd = np.array([embeddings_gd[key] for key in keys_gd_sampled])
    matrix_gpt = np.array([embeddings_gpt[key] for key in keys_gpt_sampled])
    
    # Heatmap
    heatmap_fig = plot_heatmap(matrix_gd, matrix_gpt, gpt_labels, gold_labels, keys_gpt_sampled, keys_gd_sampled)

    ########################################
    # Save both figures to a PDF
    save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, embeddings_dir)
    
    
    
    


    
    
    
    
    
    
    
