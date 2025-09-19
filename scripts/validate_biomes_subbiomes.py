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
from embeddings_functions import (
    load_embeddings, compare_embeddings, create_shuffled_background_distribution, sample_by_category
)
from stats_module import (
    calculate_overlap_and_run_tests_biomes, compare_based_on_overlap_subbiomes,
    print_statistics, test_similarity_separation
)
from output_writing import (
    plot_biome_agreement, plot_actual_vs_background, plot_heatmap,
    save_figures_to_pdf, output_to_csv
)
import argparse
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# ------------------ argument parsing ------------------
parser = argparse.ArgumentParser(
    description="Validate GPT-predicted biomes & sub-biomes."
)
parser.add_argument("--work_dir", default=".", help="Base working directory")
parser.add_argument(
    "--map_file",
    required=True,
    help="TSV with 3 columns: <filename> <label> <test_type>",
)
args = parser.parse_args()

WORK_DIR       = os.path.abspath(args.work_dir)
EMBEDDINGS_DIR = os.path.join(WORK_DIR, "embeddings")
GOLD_DICT_PATH = os.path.join(WORK_DIR, "gold_dict.pkl")

# ------------------ helpers ------------------
def to_work_path(p: str) -> str:
    """Return absolute path for p relative to WORK_DIR if p is not absolute."""
    return p if os.path.isabs(p) else os.path.join(WORK_DIR, p)

# ------------------ read TSV ------------------
map_path = os.path.join(WORK_DIR, args.map_file)

# Expect 3 columns: filename, label, test_type
df_map = pd.read_csv(
    map_path, sep="\t", comment="#", header=None,
    names=["filename", "label", "test_type"], usecols=[0, 1, 2]
)

# Normalize filenames to basenames for consistent keys/merges,
# but keep the original (possibly relative) names to load files.
df_map["filename_base"] = df_map["filename"].map(os.path.basename)

# Useful lists/maps in TSV order
my_files_base       = df_map["filename_base"].tolist()   # basenames in TSV order
my_labels           = df_map["label"].tolist()
file_label_map      = dict(zip(my_files_base, my_labels))  # basename -> label
label_to_testtype   = dict(zip(df_map["label"], df_map["test_type"]))
filename_to_testype = dict(zip(df_map["filename_base"], df_map["test_type"]))
base_to_full_path   = dict(zip(df_map["filename_base"], df_map["filename"]))

print("\nFile and its label name:\n")
for _, row in df_map.iterrows():
    print(f"{row['filename_base']} - {row['label']}\n")

# -----------------------------
# Ground truth loading & processing
# -----------------------------
with open(GOLD_DICT_PATH, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict_df = pd.DataFrame({
    'sample': list(gold_dict.keys()),
    'biome': [v[1] for v in gold_dict.values()]
})
gold_dict_json_path = os.path.join(EMBEDDINGS_DIR, 'gold_dict_sbembeddings.json')
embeddings_gd = load_embeddings(gold_dict_json_path)

# -----------------------------
# 1. Biome agreement calculation & plotting
# -----------------------------

# Load, process, and calculate agreements for data files IN TSV ORDER
full_dfs = []
for _, row in df_map.iterrows():
    fbase  = row["filename_base"]
    label  = row["label"]
    f_full = to_work_path(base_to_full_path[fbase])
    full_dfs.append(load_and_process_file(f_full, gold_dict_df, label))

full_agreement_df = pd.concat(full_dfs, ignore_index=True)
full_agreement_df['agreement'] = full_agreement_df['gpt_biome'] == full_agreement_df['biome']

lenient_agreement_df = pd.concat(full_dfs, ignore_index=True)
lenient_agreement_df['agreement'] = lenient_agreement_df.apply(
    lambda row: (
        (str(row['biome']).strip().lower() in str(row['gpt_biome']).strip().lower()
         or str(row['gpt_biome']).strip().lower() in str(row['biome']).strip().lower())
        and not pd.isna(row['biome']) and not pd.isna(row['gpt_biome'])
    ),
    axis=1
)

full_result, lenient_result = plot_biome_agreement(
    full_agreement_df, lenient_agreement_df, file_label_map, WORK_DIR
)

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

# Map index (label) -> a deterministic representative filename (first occurrence in TSV)
rep_filename_per_label = (
    df_map.drop_duplicates('label')
         .set_index('label')['filename_base']
         .to_dict()
)
results_biome['Filename'] = [rep_filename_per_label.get(label) for label in results_biome.index]

# -----------------------------
# 2. Sub-biome agreement calculation & plotting
# -----------------------------
results = {}           # filename_base -> per-sample comparison results
results_list = []      # rows for results_subbiome

# Fetch embeddings from each gpt json file and compare to ground truth embeddings IN TSV ORDER
for fbase in my_files_base:
    gpt_file_ori = fbase  # keep basename for keys/merges
    gpt_json_name = re.sub(r'\.txt|\.csv', '_sbembeddings.json', fbase)
    gpt_json_file_path = os.path.join(EMBEDDINGS_DIR, gpt_json_name)
    embeddings_gpt = load_embeddings(gpt_json_file_path)

    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    print("\nSample size after filtering:", len(filtered_gpt))

    # Compare embeddings
    compare_results = compare_embeddings(filtered_gd, filtered_gpt)

    # Calculate and print statistics
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim, std_dev, percentiles, subbiome_sample_size = print_statistics(actual_similarities)

    results[gpt_file_ori] = compare_results

    # Similarity vs background
    background_similarities = create_shuffled_background_distribution(
        filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities)
    )
    MWU_stat, MWU_p_value = test_similarity_separation(actual_similarities, background_similarities)
    title = f"Comparison of Actual vs Background Cosine Similarity for\n{gpt_json_name}"
    comparison_fig = plot_actual_vs_background(
        actual_similarities, background_similarities, title,
        avg_sim, median_sim, std_dev, MWU_stat, MWU_p_value
    )

    # Gather info for table
    results_sub_biome_dict = {
        'Average Similarity': avg_sim,
        'Median Similarity': median_sim,
        'Standard Deviation': std_dev,
        'subbiome_sample_size': subbiome_sample_size,
        '95th Percentile': percentiles,
        'MWU Statistic': MWU_stat,
        'MWU P-value': MWU_p_value,
        'Filename': gpt_file_ori,  # basename
    }
    results_list.append(results_sub_biome_dict)

    # Plotting (heatmap)
    gold_labels = {key: embeddings_gd[key]['sub-biome'] for key in embeddings_gd}
    gpt_labels = {key: embeddings_gpt[key]['sub-biome'] for key in embeddings_gpt}

    # keep only common keys in both dictionaries, and sample up to 10 per biome
    gold_biomes = {key: embeddings_gd[key]['biome'] for key in embeddings_gd}
    common_keys = list(gold_labels.keys() & gpt_labels.keys())
    sampled_keys = sample_by_category(common_keys, gold_biomes, 10)

    matrix_gd = np.array([embeddings_gd[key]['embedding'] for key in sampled_keys])
    matrix_gpt = np.array([embeddings_gpt[key]['embedding'] for key in sampled_keys])
    gold_labels_sampled = {key: gold_labels[key] for key in sampled_keys}
    gpt_labels_sampled = {key: gpt_labels[key] for key in sampled_keys}

    heatmap_fig = plot_heatmap(
        matrix_gd, matrix_gpt, gpt_labels_sampled, gold_labels_sampled, sampled_keys, sampled_keys
    )

    # Save both to a PDF
    gpt_base_file = re.sub(r'\.txt|\.csv', '', fbase)
    save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, EMBEDDINGS_DIR)

# concatenate data
results_subbiome = pd.DataFrame(results_list)

# -----------------------------
# 1. Stats for biomes (filtered to same test_type)
# -----------------------------
results_stats = calculate_overlap_and_run_tests_biomes(full_agreement_df)

# keep only label pairs from the same test_type
tt1 = results_stats['Label1'].map(label_to_testtype)
tt2 = results_stats['Label2'].map(label_to_testtype)
results_stats = results_stats[tt1 == tt2].copy()
results_stats['test_type'] = tt1[tt1 == tt2]

# deterministic representative filename per label (first occurrence)
results_stats['Filename1'] = results_stats['Label1'].map(rep_filename_per_label)
results_stats['Filename2'] = results_stats['Label2'].map(rep_filename_per_label)

results_stats['validation'] = 'biome'
print(results_stats.columns)
# colnames are: Label1 Label2 Statistic P-value Adjusted P-value Test Type Filename1 Filename2 test_type validation

# -----------------------------
# 2. Stats for sub-biomes (compare only within the same test_type)
# -----------------------------
results_data = []

# Build allowed pairs (basenames) by test_type, in TSV order
for _, grp in df_map.groupby("test_type", dropna=False, sort=False):
    files = grp["filename_base"].tolist()
    for file1, file2 in combinations(files, 2):
        print(f"\n\nComparing file:\n\n{file1}\nwith file:\n{file2}\n")
        overlap_percentage, stat, p_value, p_adjusted, test_type = compare_based_on_overlap_subbiomes(
            results[file1], results[file2]
        )
        results_data.append({
            'Statistic': stat,
            'P-value': p_value,
            'Adjusted P-value': p_adjusted,
            'Test Type': test_type,
            'Filename1': file1,
            'Filename2': file2
        })

results_df_stats = pd.DataFrame(results_data)
results_df_stats['validation'] = 'sub-biome'

# Map filename -> label for display
filename_to_label = {row.filename_base: row.label for _, row in df_map.iterrows()}
results_df_stats['Label1'] = results_df_stats['Filename1'].map(filename_to_label)
results_df_stats['Label2'] = results_df_stats['Filename2'].map(filename_to_label)

# attach test_type
results_df_stats['test_type'] = results_df_stats['Filename1'].map(filename_to_testype)

print(results_df_stats.columns)
# colnames are: Statistic P-value Adjusted P-value Test Type Filename1 Filename2 validation Label1 Label2 test_type

# -----------------------------
# Combine biome and sub-biome results (table) and enforce TSV label order
# -----------------------------
biomes_subbiomes = pd.merge(results_biome, results_subbiome, on='Filename', how='inner')
biomes_subbiomes['Label'] = biomes_subbiomes['Filename'].map({row.filename_base: row.label for _, row in df_map.iterrows()})

# Enforce label order exactly as in the TSV; and within each label keep TSV file order
label_order = df_map['label'].tolist()
file_order  = {f: i for i, f in enumerate(df_map['filename_base'].tolist())}

biomes_subbiomes['Label'] = pd.Categorical(biomes_subbiomes['Label'], categories=label_order, ordered=True)
biomes_subbiomes['__file_ord'] = biomes_subbiomes['Filename'].map(file_order)

biomes_subbiomes = (
    biomes_subbiomes
    .sort_values(['Label', '__file_ord'], kind='stable')
    .drop(columns='__file_ord')
    .reset_index(drop=True)
)

print(biomes_subbiomes)

# Write results CSV
filename = os.path.join(WORK_DIR, 'biome_subbiome_results.csv')
output_to_csv(biomes_subbiomes, filename)

# -----------------------------
# Combine biome and sub-biome stats (already filtered to same test_type)
# -----------------------------
biomes_subbiomes_stats = pd.concat([results_stats, results_df_stats], ignore_index=True)
print(biomes_subbiomes_stats.head())
print(biomes_subbiomes_stats.columns)

filename = os.path.join(WORK_DIR, 'biome_subbiome_stats.csv')
output_to_csv(biomes_subbiomes_stats, filename)
