#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:25:37 2024

@author: dgaio
"""

import os
import pickle
import pandas as pd
from gpt_files_process import find_distinguishing_features, extract_labels_from_filename
from gpt_files_process import edit_features, load_and_process_file
from plot_biome_agreement import plot_biome_agreement
from mcnemar_test_module import mcnemar_test_with_correction
import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')


# Paths
home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")

# -----------------------------
# 2. Ground truth loading & processing
# -----------------------------    
input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict_df = pd.DataFrame({
    'sample': [k for k, v in gold_dict.items()],
    'biome': [v[1] for k, v in gold_dict.items()]})

# Files
my_files = ['gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202406051335.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt']

my_files = [os.path.join(work_dir, f) for f in my_files]

features = find_distinguishing_features(my_files)
file_label_map = {file: extract_labels_from_filename(file, features) for file in my_files}
file_label_map = edit_features(file_label_map)

print("\nFile and its label name:\n")
for file, label in file_label_map.items():
    print(f"{os.path.basename(file)} - {label}\n")

# Load, process, and concatenate DataFrames
dfs = [load_and_process_file(f, gold_dict_df, label) for f, label in file_label_map.items()]
concatenated_df = pd.concat(dfs, ignore_index=True)
concatenated_df['agreement'] = concatenated_df['gpt_biome'] == concatenated_df['biome']

# Calculating agreement and plotting
plot_biome_agreement(concatenated_df, file_label_map, work_dir)


# run mcnemar test for dependent samples 
result_df = mcnemar_test_with_correction(concatenated_df)







# =============================================================================
# # For random seeds: 
# from scipy.stats import ttest_rel
# 
# # Example: calculate the t-test for the means of two related samples
# data1 = df[df['label'] == 'rs22, API118']['agreement']
# data2 = df[df['label'] == 'rs22, API119']['agreement']
# data3 = df[df['label'] == 'rs32, API118']['agreement']
# data4 = df[df['label'] == 'rs32, API120']['agreement']
# data5 = df[df['label'] == 'rs42, API109']['agreement']
# data6 = df[df['label'] == 'rs42, API110']['agreement']
# 
# # Performing the paired t-test
# stat, p = ttest_rel(data1, data2)
# print(f'Paired t-test: t={stat}, p={p}')
# stat, p = ttest_rel(data3, data4)
# print(f'Paired t-test: t={stat}, p={p}')
# stat, p = ttest_rel(data5, data6)
# print(f'Paired t-test: t={stat}, p={p}')
# 
# 
# 
# from scipy.stats import ttest_ind
# 
# # Splitting the string and keeping only the part before the underscore
# df['new_label'] = df['label'].str.split(',').str[0]
# #print(df)
# 
# group1_data = df[df['new_label'] == 'rs22']['agreement']
# group2_data = df[df['new_label'] == 'rs32']['agreement']
# group3_data = df[df['new_label'] == 'rs42']['agreement']
# 
# 
# # Performing the independent t-test
# stat, p = ttest_ind(group1_data, group2_data)
# print(f'Independent t-test: t={stat}, p={p}')
# stat, p = ttest_ind(group1_data, group3_data)
# print(f'Independent t-test: t={stat}, p={p}')
# stat, p = ttest_ind(group2_data, group3_data)
# print(f'Independent t-test: t={stat}, p={p}')
# =============================================================================

