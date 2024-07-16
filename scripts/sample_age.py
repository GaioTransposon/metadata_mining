#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:49:01 2024

@author: dgaio
"""


import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
import os
import pickle
import pandas as pd
from features_process import find_distinguishing_features, extract_labels_from_filename, edit_features, load_and_process_file
from biome_stats_module import mcnemar_test_with_correction, t_test_agreements

# -----------------------------
# Files and Paths
# -----------------------------
home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")

# List all files in the work_dir that start with 'gpt_clean' and include 'async'
my_files = [f for f in os.listdir(work_dir) if f.startswith('gpt_clean_output')and f.endswith('txt') ]   #and 'batch' in f

len(my_files)

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

# -----------------------------
# Agreement calculation & Analysis of younger samples
# ----------------------------- 
full_dfs = []
for file, label in file_label_map.items():
    # Load and process only if the sample name starts with 'SRS'
    df = load_and_process_file(file, gold_dict_df, label)
    full_dfs.append(df)

# Combine all dataframes into one
full_agreement_df = pd.concat(full_dfs, ignore_index=True)
full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']

# Filter to include only samples starting with 'SRS'
srs_agreement_df = full_agreement_df[full_agreement_df['sample'].str.startswith('ERS')]

# Extract the numeric part from the 'SRS' sample identifiers and convert to integer
srs_agreement_df['sample_numeric'] = srs_agreement_df['sample'].str.extract('ERS(\d+)').astype(int)

print(srs_agreement_df)

print(len(srs_agreement_df))




# Sort the DataFrame by the numeric sample identifier
srs_agreement_df_sorted = srs_agreement_df.sort_values(by='sample_numeric')

# Optional: Create age groups (quartiles) based on the numeric identifiers for more detailed analysis
srs_agreement_df_sorted['age_quartile'] = pd.qcut(srs_agreement_df_sorted['sample_numeric'], 4, labels=["oldest", "older", "younger", "youngest"])

# Calculate the mean agreement in each quartile
agreement_by_age_group = srs_agreement_df_sorted.groupby('age_quartile')['agreement'].mean()

print("Agreement by age quartile:")
print(agreement_by_age_group)

