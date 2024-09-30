#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:50:30 2024

@author: dgaio
"""


# this script is run by run_validate_biomes_subbiomes.sh 

import os
import pandas as pd
import pickle
import numpy as np
import sys
import re
from itertools import combinations
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from features_process import find_distinguishing_features, extract_labels_from_filename, edit_features, load_and_process_file, filter_common_keys
from embeddings_functions import (load_embeddings, compare_embeddings, create_shuffled_background_distribution, sample_by_category)
from stats_module import calculate_overlap_and_run_tests_biomes, compare_based_on_overlap_subbiomes, print_statistics, test_similarity_separation
from output_writing import plot_biome_agreement, plot_actual_vs_background, plot_heatmap, save_figures_to_pdf, output_to_csv
import argparse

# -----------------------------
# Files and Paths
# ----------------------------- 

home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")
embeddings_dir = os.path.join(work_dir, "embeddings")
gold_dict_path = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")


# -----------------------------
# Ground truth loading & processing
# -----------------------------   
with open(gold_dict_path, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict_df = pd.DataFrame({'sample': list(gold_dict.keys()), 'biome': [v[1] for v in gold_dict.values()]})
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')
embeddings_gd = load_embeddings(gold_dict_json_path)


# -----------------------------
# Files processing
# ----------------------------- 
    
parser = argparse.ArgumentParser(description='Process files and labels.')

# Add arguments for files and labels. Expecting a list for each.
parser.add_argument('--files', nargs='+', help='List of files', required=True)
parser.add_argument('--labels', nargs='+', help='List of labels for the files', required=True)

# Parse the arguments
args = parser.parse_args()

# Assign files and labels from arguments
my_files = args.files
my_labels = args.labels

file_label_map = dict(zip(my_files, my_labels))

print("\nFile and its label name:\n")
for file, label in file_label_map.items():
    print(f"{os.path.basename(file)} - {label}\n")

    
# -----------------------------
# 1. Biome agreement calculation & plotting 
# ----------------------------- 

# Load, process, and calculate agreements for data files
full_dfs = [load_and_process_file(os.path.join(work_dir, f), gold_dict_df, label) for f, label in file_label_map.items()]
full_agreement_df = pd.concat(full_dfs, ignore_index=True)
full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']
lenient_agreement_df = pd.concat(full_dfs, ignore_index=True)
#lenient_agreement_df['agreement'] = lenient_agreement_df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)

lenient_agreement_df['agreement'] = lenient_agreement_df.apply(
    lambda row: ((str(row['biome']).strip().lower() in str(row['gpt_biome']).strip().lower() or
                 str(row['gpt_biome']).strip().lower() in str(row['biome']).strip().lower()) and
                 not pd.isna(row['biome']) and not pd.isna(row['gpt_biome'])),
    axis=1
)


full_result, lenient_result = plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir)

results_biome = pd.concat([
    full_result[['full match label']].rename(columns={'full match label': 'Agreement biome (exact match)'}),
    lenient_result[['full+partial match label']].rename(columns={'full+partial match label': 'Agreement biome (lenient match)'})
], axis=1)

filename_label_map = {label: os.path.basename(file) for file, label in file_label_map.items()}
results_biome['Filename'] = [filename_label_map.get(label) for label in results_biome.index]


# -----------------------------
# 2. Sub-biome agreement calculation & plotting 
# ----------------------------- 

results={}
results_sub_biome = []
results_list = []

# Fetch embeddings from each gpt json file and compare to ground truth embeddings:
for gpt_file in my_files:    
    
    gpt_file_ori = gpt_file
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
    
    results[gpt_file_ori] = compare_results

    
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
        'Average Similarity': avg_sim,
        'Median Similarity': median_sim,
        'Standard Deviation': std_dev,
        '95th Percentile': percentiles,  
        'MWU Statistic': MWU_stat,
        'MWU P-value': MWU_p_value,
        'Filename': gpt_file_ori,
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
results_subbiome = pd.DataFrame(results_list)



# -----------------------------
# 1. Stats for biomes
# ----------------------------- 

results_stats = calculate_overlap_and_run_tests_biomes(full_agreement_df) 

results_stats['Filename1'] = results_stats['Label1'].map(filename_label_map)
results_stats['Filename2'] = results_stats['Label2'].map(filename_label_map)

results_stats['validation'] = 'biome'
print(results_stats.columns)
# colnames are: 	Label1	Label2	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2


# -----------------------------
# 2. Stats for sub-biomes
# ----------------------------- 

results_data = []

# Stats pairwise comparisons with dynamic test selection based on overlap (if more than 70% samples in common, dependent test)
for file1, file2 in combinations(results.keys(), 2):
    print(f"\n\nComparing file:\n\n{file1}\nwith file:\n{file2}\n")
    overlap_percentage, stat, p_value, p_adjusted, test_type = compare_based_on_overlap_subbiomes(results[file1], results[file2])

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

# key to value reverse
reversed_filename_label_map = {v: k for k, v in filename_label_map.items()}
results_df_stats['Label1'] = results_df_stats['Filename1'].map(reversed_filename_label_map)
results_df_stats['Label2'] = results_df_stats['Filename2'].map(reversed_filename_label_map)

print(results_df_stats.columns)
# colnames are: 	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2







# Combine biome and sub-biome results: 
biomes_subbiomes = pd.merge(results_biome, results_subbiome, on='Filename', how='inner')
biomes_subbiomes['Label'] = biomes_subbiomes['Filename'].map(file_label_map)
print(biomes_subbiomes)


filename = os.path.join(work_dir, 'biome_subbiome_results.csv')
output_to_csv(biomes_subbiomes, filename)



# Combine biome and sub-biome stats 
biomes_subbiomes_stats = pd.concat([results_stats, results_df_stats], ignore_index=True)
print(biomes_subbiomes_stats.head())
print(biomes_subbiomes_stats.columns)



filename = os.path.join(work_dir, 'biome_subbiome_stats.csv')
output_to_csv(biomes_subbiomes_stats, filename)






# =============================================================================
# import os
# import pandas as pd
# import pickle
# import numpy as np
# import sys
# import re
# from itertools import combinations
# sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
# from features_process import find_distinguishing_features, extract_labels_from_filename, edit_features, load_and_process_file, filter_common_keys
# from embeddings_functions import (load_embeddings, compare_embeddings, create_shuffled_background_distribution, sample_by_category)
# from stats_module import calculate_overlap_and_run_tests_biomes, compare_based_on_overlap_subbiomes, print_statistics, test_similarity_separation
# from output_writing import plot_biome_agreement, plot_actual_vs_background, plot_heatmap, save_figures_to_pdf, output_to_csv
# 
# 
# 
# 
# # -----------------------------
# # Files and Paths
# # ----------------------------- 
# 
# home_dir = os.getenv('HOME')
# work_dir = os.path.join(home_dir, "MicrobeAtlasProject")
# embeddings_dir = os.path.join(work_dir, "embeddings")
# gold_dict_path = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
# 
# 
# # -----------------------------
# # Ground truth loading & processing
# # -----------------------------   
# with open(gold_dict_path, 'rb') as file:
#     gold_dict = pickle.load(file)
# gold_dict_df = pd.DataFrame({'sample': list(gold_dict.keys()), 'biome': [v[1] for v in gold_dict.values()]})
# gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')
# embeddings_gd = load_embeddings(gold_dict_json_path)
# 
# 
# # -----------------------------
# # Files processing
# # ----------------------------- 
# 
# features = find_distinguishing_features(my_files)
# file_label_map = {file: extract_labels_from_filename(file, features) for file in my_files}
# file_label_map = edit_features(file_label_map)
# 
# print("\nFile and its label name:\n")
# for file, label in file_label_map.items():
#     print(f"{os.path.basename(file)} - {label}\n")
#     
#     
#     
# # -----------------------------
# # 1. Biome agreement calculation & plotting 
# # ----------------------------- 
# 
# # Load, process, and calculate agreements for data files
# full_dfs = [load_and_process_file(os.path.join(work_dir, f), gold_dict_df, label) for f, label in file_label_map.items()]
# full_agreement_df = pd.concat(full_dfs, ignore_index=True)
# full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']
# lenient_agreement_df = pd.concat(full_dfs, ignore_index=True)
# #lenient_agreement_df['agreement'] = lenient_agreement_df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)
# 
# lenient_agreement_df['agreement'] = lenient_agreement_df.apply(
#     lambda row: ((str(row['biome']).strip().lower() in str(row['gpt_biome']).strip().lower() or
#                  str(row['gpt_biome']).strip().lower() in str(row['biome']).strip().lower()) and
#                  not pd.isna(row['biome']) and not pd.isna(row['gpt_biome'])),
#     axis=1
# )
# 
# 
# full_result, lenient_result = plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir)
# 
# results_biome = pd.concat([
#     full_result[['full match label']].rename(columns={'full match label': 'Agreement biome (exact match)'}),
#     lenient_result[['full+partial match label']].rename(columns={'full+partial match label': 'Agreement biome (lenient match)'})
# ], axis=1)
# 
# filename_label_map = {label: os.path.basename(file) for file, label in file_label_map.items()}
# results_biome['Filename'] = [filename_label_map.get(label) for label in results_biome.index]
# 
# 
# # -----------------------------
# # 2. Sub-biome agreement calculation & plotting 
# # ----------------------------- 
# 
# results={}
# results_sub_biome = []
# results_list = []
# 
# # Fetch embeddings from each gpt json file and compare to ground truth embeddings:
# for gpt_file in my_files:    
#     
#     gpt_file_ori = gpt_file
#     gpt_file = re.sub(r'\.txt|\.csv', '_sbembeddings.json', gpt_file)
#     gpt_json_file_path = os.path.join(embeddings_dir, gpt_file) 
#     embeddings_gpt = load_embeddings(gpt_json_file_path)
#     
#     # Filter embeddings to include only common keys
#     filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
#     print("\nSample size after filtering:", len(filtered_gpt))
# 
#     ########################################
#     # Compare embeddings
#     compare_results = compare_embeddings(filtered_gd, filtered_gpt)
# 
#     ########################################
#     # Calculate and print statistics
#     actual_similarities = [result['cosine'] for result in compare_results.values()]
#     avg_sim, median_sim, std_dev, percentiles = print_statistics(actual_similarities)
#     
#     results[gpt_file_ori] = compare_results
# 
#     
#     ########################################
#     # Similarity vs background
#     background_similarities = create_shuffled_background_distribution(filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities))
#     MWU_stat, MWU_p_value = test_similarity_separation(actual_similarities, background_similarities)
#     title = f"Comparison of Actual vs Background Cosine Similarity for\n{gpt_file}"
#     comparison_fig = plot_actual_vs_background(actual_similarities, background_similarities, title, 
#                                                avg_sim, median_sim, std_dev,
#                                                MWU_stat, MWU_p_value)
#     
#     ########################################
#     # Gather info: 
#     #avg_sim, median_sim, std_dev, percentiles, MWU_stat, MWU_p_value, filename, label
#     results_sub_biome = {
#         'Average Similarity': avg_sim,
#         'Median Similarity': median_sim,
#         'Standard Deviation': std_dev,
#         '95th Percentile': percentiles,  
#         'MWU Statistic': MWU_stat,
#         'MWU P-value': MWU_p_value,
#         'Filename': gpt_file_ori,
#     }
#     
#     # Append the dictionary to the results list
#     results_list.append(results_sub_biome)
#     
#     ########################################
#     # Plotting: 
#         
#     # Extract sub_biome_texts directly from embeddings dictionaries
#     gold_labels = {key: embeddings_gd[key]['sub-biome'] for key in embeddings_gd}
#     gpt_labels = {key: embeddings_gpt[key]['sub-biome'] for key in embeddings_gpt}
#     
#     ### filtering to keep only common keys in both dictionaries, and 10 per biome
#     gold_biomes = {key: embeddings_gd[key]['biome'] for key in embeddings_gd}
#     common_keys = list(gold_labels.keys() & gpt_labels.keys())
#     sampled_keys = sample_by_category(common_keys, gold_biomes, 10)
#     ###
# 
#     # Prepare data matrices for the heatmap
#     matrix_gd = np.array([embeddings_gd[key]['embedding'] for key in sampled_keys])
#     matrix_gpt = np.array([embeddings_gpt[key]['embedding'] for key in sampled_keys])
#     gold_labels_sampled = {key: gold_labels[key] for key in sampled_keys}
#     gpt_labels_sampled = {key: gpt_labels[key] for key in sampled_keys}
# 
#     # Generate heatmap
#     heatmap_fig = plot_heatmap(matrix_gd, matrix_gpt, gpt_labels_sampled, gold_labels_sampled, sampled_keys, sampled_keys)
# 
#     ########################################
#     # Save both to a PDF
#     gpt_base_file = gpt_file.replace('_sbembeddings.json', '')
#     save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, embeddings_dir)
# 
#     
# # concatenate data 
# results_subbiome = pd.DataFrame(results_list)
# 
# 
# 
# # -----------------------------
# # 1. Stats for biomes
# # ----------------------------- 
# 
# results_stats = calculate_overlap_and_run_tests_biomes(full_agreement_df) 
# 
# results_stats['Filename1'] = results_stats['Label1'].map(filename_label_map)
# results_stats['Filename2'] = results_stats['Label2'].map(filename_label_map)
# 
# results_stats['validation'] = 'biome'
# print(results_stats.columns)
# # colnames are: 	Label1	Label2	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2
# 
# 
# # -----------------------------
# # 2. Stats for sub-biomes
# # ----------------------------- 
# 
# results_data = []
# 
# # Stats pairwise comparisons with dynamic test selection based on overlap (if more than 70% samples in common, dependent test)
# for file1, file2 in combinations(results.keys(), 2):
#     print(f"\n\nComparing file:\n\n{file1}\nwith file:\n{file2}\n")
#     overlap_percentage, stat, p_value, p_adjusted, test_type = compare_based_on_overlap_subbiomes(results[file1], results[file2])
# 
#     result_dict = {
#         # label1 and labels2
#         'Statistic': stat,
#         'P-value': p_value,
#         'Adjusted P-value': p_adjusted,
#         'Test Type': test_type,
#         'Filename1': file1,
#         'Filename2': file2
#     }
# 
#     results_data.append(result_dict)
#     
# results_df_stats = pd.DataFrame(results_data)
# results_df_stats['validation'] = 'sub-biome'
# 
# # key to value reverse
# reversed_filename_label_map = {v: k for k, v in filename_label_map.items()}
# results_df_stats['Label1'] = results_df_stats['Filename1'].map(reversed_filename_label_map)
# results_df_stats['Label2'] = results_df_stats['Filename2'].map(reversed_filename_label_map)
# 
# print(results_df_stats.columns)
# # colnames are: 	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2
# 
# 
# 
# 
# 
# 
# 
# # Combine biome and sub-biome results: 
# biomes_subbiomes = pd.merge(results_biome, results_subbiome, on='Filename', how='inner')
# biomes_subbiomes['Label'] = biomes_subbiomes['Filename'].map(file_label_map)
# print(biomes_subbiomes)
# 
# 
# filename = os.path.join(work_dir, 'biome_subbiome_results.csv')
# output_to_csv(biomes_subbiomes, filename)
# 
# 
# 
# # Combine biome and sub-biome stats 
# biomes_subbiomes_stats = pd.concat([results_stats, results_df_stats], ignore_index=True)
# print(biomes_subbiomes_stats.head())
# print(biomes_subbiomes_stats.columns)
# 
# 
# 
# filename = os.path.join(work_dir, 'biome_subbiome_stats.csv')
# output_to_csv(biomes_subbiomes_stats, filename)
# =============================================================================
