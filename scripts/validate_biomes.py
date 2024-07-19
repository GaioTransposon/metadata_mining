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

# -----------------------------
# Files processing
# ----------------------------- 

features = find_distinguishing_features(my_files)
file_label_map = {file: extract_labels_from_filename(file, features) for file in my_files}
file_label_map = edit_features(file_label_map)

print("\nFile and its label name:\n")
for file, label in file_label_map.items():
    print(f"{os.path.basename(file)} - {label}\n")


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
# Biome agreement calculation
# ----------------------------- 

full_dfs = [
    load_and_process_file(os.path.join(work_dir, f), gold_dict_df, label)
    for f, label in file_label_map.items()
]
full_agreement_df = pd.concat(full_dfs, ignore_index=True)
full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']

lenient_agreement_df = pd.concat(full_dfs, ignore_index=True)
lenient_agreement_df['agreement'] = lenient_agreement_df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)

# -----------------------------
# Biome agreement plotting
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
# colnames are: label	Agreement biome (exact match)	Agreement biome (lenient match)	Filename



# -----------------------------
# Stats
# ----------------------------- 

results_stats = calculate_overlap_and_run_tests(full_agreement_df) 

results_stats['Filename1'] = results_stats['Label1'].map(filename_label_map)
results_stats['Filename2'] = results_stats['Label2'].map(filename_label_map)

results_stats['validation'] = 'biome'
print(results_stats)
# colnames are: 	Label1	Label2	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2





# validate_subbiomes needs the Label1 and Label2 columns

# combine validate_biomes.py and validate_subbiomes.py stats part

# output unique csv for biome+sub-biome

# output unique csv for biome stats and sub-biome stats

# make sure csv if populated at each loop by first jumping 1 row when concatenating 








