#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:25:37 2024

@author: dgaio
"""

import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
import os
import pickle
import pandas as pd
from features_process import find_distinguishing_features, extract_labels_from_filename, edit_features, load_and_process_file
from plot_biome_agreement import lenient_match, plot_biome_agreement
from biome_stats_module import calculate_overlap_and_run_tests

# -----------------------------
# Files and Paths
# ----------------------------- 
home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")

# # my_files is in middle_dir/my_files.txt
# # expected different
# my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
#             'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']

# -----------------------------
# Ground truth loading & processing
# -----------------------------    
input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict_df = pd.DataFrame({
    'sample': [k for k, v in gold_dict.items()],
    'biome': [v[1] for k, v in gold_dict.items()]})

# -----------------------------
# Files processing
# ----------------------------- 
my_files = [os.path.join(work_dir, f) for f in my_files]    
features = find_distinguishing_features(my_files)
file_label_map = {file: extract_labels_from_filename(file, features) for file in my_files}
file_label_map = edit_features(file_label_map)

print("\nFile and its label name:\n")
for file, label in file_label_map.items():
    print(f"{os.path.basename(file)} - {label}\n")


# -----------------------------
# Agreement calculation
# ----------------------------- 
full_dfs = [load_and_process_file(f, gold_dict_df, label) for f, label in file_label_map.items()]
full_agreement_df = pd.concat(full_dfs, ignore_index=True)
full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']

lenient_dfs = [load_and_process_file(f, gold_dict_df, label) for f, label in file_label_map.items()]
lenient_agreement_df = pd.concat(lenient_dfs, ignore_index=True)
lenient_agreement_df['agreement'] = lenient_agreement_df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)

# -----------------------------
# Plotting
# ----------------------------- 
full_result, lenient_result = plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir)

# Combining the full and lenient results into a single DataFrame
full_result, lenient_result = plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir)
combined_results = pd.concat([
    full_result[['full match label']].rename(columns={
        'full match label': 'Agreement biome (exact match)'
    }),
    lenient_result[['full+partial match label']].rename(columns={
        'full+partial match label': 'Agreement biome (lenient match)'
    })
], axis=1)

filename_label_map = {label: os.path.basename(file) for file, label in file_label_map.items()}
combined_results['Filename'] = [filename_label_map.get(label) for label in combined_results.index]

print(combined_results)

# -----------------------------
# Stats
# ----------------------------- 

results_stats = calculate_overlap_and_run_tests(full_agreement_df) 

results_stats['Filename1'] = results_stats['Label1'].map(filename_label_map)
results_stats['Filename2'] = results_stats['Label2'].map(filename_label_map)

print(results_stats)









