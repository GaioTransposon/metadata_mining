#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 23:28:02 2024

@author: dgaio
"""


import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
import os
import pandas as pd
import numpy as np
import pickle
import re
from itertools import combinations
import matplotlib.pyplot as plt
from features_process import find_distinguishing_features, extract_labels_from_filename, edit_features, load_and_process_file, filter_common_keys
from plot_biome_agreement import lenient_match, plot_biome_agreement
from biome_stats_module import calculate_overlap_and_run_tests
from embeddings_functions import (load_embeddings, compare_embeddings, print_statistics, create_shuffled_background_distribution, sample_by_category, plot_heatmap, save_figures_to_pdf, plot_actual_vs_background)
from embeddings_functions import (load_embeddings, compare_embeddings, print_statistics,
                                  create_shuffled_background_distribution,
                                  sample_by_category, plot_heatmap, save_figures_to_pdf,
                                  plot_actual_vs_background,
                                  test_similarity_separation, compare_based_on_overlap)


# Common Directories and Files
home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'

# Load Gold Dictionary
input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict_df = pd.DataFrame({
    'sample': [k for k, v in gold_dict.items()],
    'biome': [v[1] for k, v in gold_dict.items()]})


# Files processing
features = find_distinguishing_features(my_files)
file_label_map = {file: extract_labels_from_filename(file, features) for file in my_files}
file_label_map = edit_features(file_label_map)


results_list = []
for file, label in file_label_map.items():
    # Biome agreement calculation
    full_dfs = load_and_process_file(os.path.join(work_dir, file), gold_dict_df, label)
    full_agreement_df = pd.DataFrame(full_dfs)  # Assuming full_dfs is a list of dataframes
    full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']
    lenient_agreement_df = full_agreement_df.copy()
    lenient_agreement_df['agreement'] = lenient_agreement_df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)

    # Obtain plotting results
    full_result, lenient_result = plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir)
    combined_results = pd.concat([full_result, lenient_result], axis=1)


    # Embeddings comparison
    gpt_file = re.sub(r'\.txt|\.csv', '_sbembeddings.json', file)
    gpt_json_file_path = os.path.join(embeddings_dir, gpt_file)
    embeddings_gpt = load_embeddings(gpt_json_file_path)
    embeddings_gd = load_embeddings(os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json'))  # Update as necessary

    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    compare_results = compare_embeddings(filtered_gd, filtered_gpt)
    
    ########################################
    # Calculate and print statistics
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim, std_dev, percentiles = print_statistics(actual_similarities)
    
    
    ########################################
    # Similarity vs background
    background_similarities = create_shuffled_background_distribution(filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities))
    MWU_stat, MWU_p_value = test_similarity_separation(actual_similarities, background_similarities)
    title = f"Comparison of Actual vs Background Cosine Similarity for\n{gpt_file}"
    comparison_fig = plot_actual_vs_background(actual_similarities, background_similarities, title, 
                                               avg_sim, median_sim, std_dev,
                                               MWU_stat, MWU_p_value)
    
    ########################################

    # Append results for DataFrame
    results_sub_biome = {
        'Label': label,
        'Average Similarity': avg_sim,
        'Median Similarity': median_sim,
        'Standard Deviation': std_dev,
        '95th Percentile': percentiles,
        'MWU Statistic': MWU_stat,
        'MWU P-value': MWU_p_value,
        'Filename': os.path.basename(file),
        'Agreement biome (exact match)': combined_results.loc[label, 'full match label'],
        'Agreement biome (lenient match)': combined_results.loc[label, 'full+partial match label']
    }
    results_list.append(results_sub_biome)

# Create final DataFrame
results_df = pd.DataFrame(results_list)
print(results_df)









combined_results = pd.concat([results_stats, results_df_stats])
print(combined_results)





