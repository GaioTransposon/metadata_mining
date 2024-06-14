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
from biome_stats_module import mcnemar_test_with_correction
from biome_stats_module import t_test_agreements
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
# my_files = ['gpt_clean_output_nspb100_chunkingno_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API499_normal_dt202406051335.txt',
#             'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt']

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

# run independent t-test for indipendent samples 
result_df = t_test_agreements(concatenated_df)


