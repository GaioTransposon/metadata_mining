#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:50:30 2024

@author: dgaio
"""


# run as: 
# python ~/github/metadata_mining/scripts/validate_biomes_subbiomes.py \
#   --work_dir ~/MicrobeAtlasProject \
#   --map_file gpt_file_label_map.tsv


import os
import pandas as pd
import pickle
import numpy as np
import re
from itertools import combinations
from features_process import load_and_process_file, filter_common_keys
from embeddings_functions import (load_embeddings, compare_embeddings, create_shuffled_background_distribution, sample_by_category)
from stats_module import calculate_overlap_and_run_tests_biomes, compare_based_on_overlap_subbiomes, print_statistics, test_similarity_separation
from output_writing import plot_biome_agreement, plot_actual_vs_background, plot_heatmap, save_figures_to_pdf, output_to_csv
import argparse
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)




# ------------------ NEW argument parsing ------------------
parser = argparse.ArgumentParser(
    description="Validate GPT-predicted biomes & sub-biomes."
)
parser.add_argument("--work_dir", default=".", help="Base working directory")
parser.add_argument(
    "--map_file",
    required=True,
    help="TSV with at least two columns: <filename> <label> "
         "(optional 3rd column like test_type is ignored)",
)
args = parser.parse_args()

WORK_DIR       = os.path.abspath(args.work_dir)
EMBEDDINGS_DIR = os.path.join(WORK_DIR, "embeddings")
GOLD_DICT_PATH = os.path.join(WORK_DIR, "gold_dict.pkl")


# ------------- read TSV -------------
map_path = os.path.join(WORK_DIR, args.map_file)

# CHANGED: read the 3rd column as test_type so we can group stats by it
df_map = pd.read_csv(
    map_path, sep="\t", comment="#", header=None,
    names=["filename", "label", "test_type"], usecols=[0, 1, 2]
)

my_files       = df_map["filename"].tolist()
my_labels      = df_map["label"].tolist()
file_label_map = dict(zip(my_files, my_labels))
# CHANGED: filename order map (TSV order) to re-apply final ordering
_file_order = {fn: i for i, fn in enumerate(my_files)}

print("\nFile and its label name:\n")
for file, label in file_label_map.items():
    print(f"{os.path.basename(file)} - {label}\n")



# -----------------------------
# Ground truth loading & processing
# -----------------------------   
with open(GOLD_DICT_PATH, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict_df = pd.DataFrame({'sample': list(gold_dict.keys()), 'biome': [v[1] for v in gold_dict.values()]})
gold_dict_json_path = os.path.join(EMBEDDINGS_DIR, 'gold_dict_sbembeddings.json')
embeddings_gd = load_embeddings(gold_dict_json_path)


    
# -----------------------------
# 1. Biome agreement calculation & plotting 
# ----------------------------- 

# Load, process, and calculate agreements for data files
full_dfs = [load_and_process_file(os.path.join(WORK_DIR, f), gold_dict_df, label) for f, label in file_label_map.items()]
full_agreement_df = pd.concat(full_dfs, ignore_index=True)
full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']
lenient_agreement_df = pd.concat(full_dfs, ignore_index=True)

lenient_agreement_df['agreement'] = lenient_agreement_df.apply(
    lambda row: ((str(row['biome']).strip().lower() in str(row['gpt_biome']).strip().lower() or
                 str(row['gpt_biome']).strip().lower() in str(row['biome']).strip().lower()) and
                 not pd.isna(row['biome']) and not pd.isna(row['gpt_biome'])),
    axis=1
)


full_result, lenient_result = plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, WORK_DIR)


results_biome = pd.concat([
    # full match 
    full_result[['full match label']].rename(columns={'full match label': 'Agreement biome (exact match)'}),
    full_result[['mean']].rename(columns={'mean': 'biome_exact_match_mean'}),
    full_result[['sd']].rename(columns={'sd': 'biome_exact_match_sd'}),

    # lenient match
    lenient_result[['full+partial match label']].rename(columns={'full+partial match label': 'Agreement biome (lenient match)'}),
    lenient_result[['mean']].rename(columns={'mean': 'biome_lenient_match_mean'}),
    lenient_result[['sd']].rename(columns={'sd': 'biome_lenient_match_sd'}),
    
    # in common
    full_result[['Full Total Counts']].rename(columns={'Full Total Counts': 'sample_size'})
], axis=1)

# CHANGED: keep label explicit for merging later (per-file results will join on label)
results_biome = results_biome.rename_axis('Label').reset_index()


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
    gpt_json_file_path = os.path.join(EMBEDDINGS_DIR, gpt_file) 
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
    avg_sim, median_sim, std_dev, percentiles, subbiome_sample_size = print_statistics(actual_similarities)

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
    results_sub_biome_dict = {
    'Average Similarity': avg_sim,
    'Median Similarity': median_sim,
    'Standard Deviation': std_dev,
    'subbiome_sample_size': subbiome_sample_size,
    '95th Percentile': percentiles,  
    'MWU Statistic': MWU_stat,
    'MWU P-value': MWU_p_value,
    'Filename': gpt_file_ori,
}
    
    # Append the dictionary to the results list
    results_list.append(results_sub_biome_dict)
    
    

    
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
    save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, EMBEDDINGS_DIR)

    
# concatenate data 
results_subbiome = pd.DataFrame(results_list)

# CHANGED: add labels to per-file sub-biome results for merging
results_subbiome['Label'] = results_subbiome['Filename'].map(file_label_map)



# -----------------------------
# 1. Stats for biomes
# ----------------------------- 

results_stats = calculate_overlap_and_run_tests_biomes(full_agreement_df) 

# CHANGED: filter biome stats to label pairs within the same test_type
_label_to_testtype = dict(zip(df_map['label'], df_map['test_type']))
_tt1 = results_stats['Label1'].map(_label_to_testtype)
_tt2 = results_stats['Label2'].map(_label_to_testtype)
results_stats = results_stats[_tt1 == _tt2].copy()
results_stats['test_type'] = _tt1[_tt1 == _tt2]

# CHANGED: map labels to a representative filename (first occurrence in TSV), for display
_rep_filename_per_label = df_map.drop_duplicates('label').set_index('label')['filename'].to_dict()
results_stats['Filename1'] = results_stats['Label1'].map(_rep_filename_per_label)
results_stats['Filename2'] = results_stats['Label2'].map(_rep_filename_per_label)

results_stats['validation'] = 'biome'
print(results_stats.columns)
# colnames are: 	Label1	Label2	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2 test_type validation


# -----------------------------
# 2. Stats for sub-biomes
# ----------------------------- 

results_data = []

# CHANGED: compare only files within the same test_type (grouping from TSV)
for _, grp in df_map.groupby('test_type', sort=False):
    files = grp['filename'].tolist()
    for file1, file2 in combinations(files, 2):
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

# CHANGED: map filename -> label directly (no reversing needed)
results_df_stats['Label1'] = results_df_stats['Filename1'].map(file_label_map)
results_df_stats['Label2'] = results_df_stats['Filename2'].map(file_label_map)

print(results_df_stats.columns)
# colnames are: 	Statistic	P-value	Adjusted P-value	Test Type	Filename1	Filename2 validation Label1 Label2




# Combine biome and sub-biome results: 
# CHANGED: merge ON 'Label' so every file row is kept (many files can share a label)
biomes_subbiomes = pd.merge(results_subbiome, results_biome, on='Label', how='left')
print(biomes_subbiomes)

# CHANGED: enforce TSV file order (unique filenames) so output rows match the TSV sequence
biomes_subbiomes['__ord'] = biomes_subbiomes['Filename'].map(_file_order)
biomes_subbiomes = (
    biomes_subbiomes
    .sort_values('__ord', kind='stable')
    .drop(columns='__ord')
    .reset_index(drop=True)
)

filename = os.path.join(WORK_DIR, 'biome_subbiome_results.csv')
output_to_csv(biomes_subbiomes, filename)



# Combine biome and sub-biome stats 
biomes_subbiomes_stats = pd.concat([results_stats, results_df_stats], ignore_index=True)
print(biomes_subbiomes_stats.head())
print(biomes_subbiomes_stats.columns)



filename = os.path.join(WORK_DIR, 'biome_subbiome_stats.csv')
output_to_csv(biomes_subbiomes_stats, filename)
