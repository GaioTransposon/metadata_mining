#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:29:00 2025

@author: danielagaio
"""



import pandas as pd
import re
from tqdm import tqdm
import random


# Load data
gpt_biomes_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt'
mapping_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/sample_taxid_mapping_clean_fixed_matlas2024.tsv'
gpt_sub_biomes_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_sub_biomes.txt'

# Read GPT_biomes.txt
df_biomes = pd.read_csv(gpt_biomes_fp, sep='\t', header=None, names=['sample_id', 'biome'])
df_biomes_filtered = df_biomes[df_biomes['biome'].isin(['animal', 'plant'])]

# Read sample_taxid_mapping_clean_fixed_matlas2024.tsv
df_mapping = pd.read_csv(mapping_fp, sep='\t')

# Filter both dataframes by sample IDs
sample_ids = df_biomes_filtered['sample_id']
df_mapping_filtered = df_mapping[df_mapping.iloc[:, 0].isin(sample_ids)]

# Merge columns from 2nd column onwards into a single column, space-separated
df_mapping_filtered = df_mapping[df_mapping.iloc[:, 0].isin(sample_ids)].copy()
df_mapping_filtered['merged'] = df_mapping_filtered.iloc[:, 1:].astype(str).apply(lambda row: ' '.join(row), axis=1)


# Read GPT_sub_biomes.txt
df_sub_biomes = pd.read_csv(gpt_sub_biomes_fp, sep='\t', header=None, names=['sample_id', 'sub_biome'])

# Filter sub-biomes by sample ids
df_sub_biomes_filtered = df_sub_biomes[df_sub_biomes['sample_id'].isin(sample_ids)]








# Clean dataframe copies
df_mapping_filtered = df_mapping[df_mapping.iloc[:, 0].isin(sample_ids)].copy()
df_mapping_filtered['merged'] = df_mapping_filtered.iloc[:, 1:].astype(str).apply(lambda row: ' '.join(row), axis=1)
df_mapping_filtered['merged'] = df_mapping_filtered['merged'].str.replace('species', '', case=False, regex=False)

df_sub_biomes_filtered = df_sub_biomes[df_sub_biomes['sample_id'].isin(sample_ids)].copy()
df_sub_biomes_filtered['sub_biome'] = df_sub_biomes_filtered['sub_biome'].str.replace('species', '', case=False, regex=False)

# Create dict of sample_id -> merged_text
sample_to_merged = df_mapping_filtered.dropna(subset=['merged']).set_index(df_mapping_filtered.columns[0])['merged'].astype(str).to_dict()

results_full = []
results_partial = []

# Iterate over sub-biomes with sample IDs
for _, row in tqdm(df_sub_biomes_filtered.iterrows(), total=len(df_sub_biomes_filtered), desc="Sample-wise matching"):
    sample_id = row['sample_id']
    sub_biome = row['sub_biome']
    
    # Safeguard against NaN or wrong data
    if not isinstance(sub_biome, str) or sample_id not in sample_to_merged:
        continue
    
    merged_text = sample_to_merged[sample_id]
    
    # Full word boundary match
    pattern_full = r'\b' + re.escape(sub_biome.strip()) + r'\b'
    if re.search(pattern_full, merged_text):
        results_full.append((sample_id, sub_biome, merged_text))
        continue  # If full match, skip partial check to avoid duplication
    
    # Partial match: any word from sub_biome exists as whole word in merged_text
    sub_biome_words = sub_biome.strip().split()
    for word in sub_biome_words:
        pattern_word = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern_word, merged_text):
            results_partial.append((sample_id, sub_biome, merged_text))
            break  # only need one word to match to consider as partial

# Deduplicate
results_full = list(set(results_full))
results_partial = list(set(results_partial))

# Clean, readable printing
def print_readable_samples(results, title, n=30):
    print(f"\n=== {title} ===")
    if not results:
        print("No matches found.\n")
        return
    sample_results = random.sample(results, min(n, len(results)))
    for i, (sample_id, sub_biome, merged) in enumerate(sample_results, 1):
        print(f"{i}. SAMPLE ID: {sample_id}\n   SUB-BIOME: {sub_biome}\n   MERGED TEXT: {merged}\n{'-'*80}")

print_readable_samples(results_full, "FULL MATCHES (whole word match only)")
print_readable_samples(results_partial, "PARTIAL MATCHES (any word match)")

# Optionally print totals
print(f"\nTotal FULL matches: {len(results_full)}")
print(f"Total PARTIAL matches: {len(results_partial)}")






# Track what was matched (for sample-specific checking)
matched_full_set = set((sample_id, sub_biome) for sample_id, sub_biome, _ in results_full)
matched_partial_set = set((sample_id, sub_biome) for sample_id, sub_biome, _ in results_partial)

# All attempted combinations (from df_sub_biomes_filtered, for consistency)
all_attempts_set = set(zip(df_sub_biomes_filtered['sample_id'], df_sub_biomes_filtered['sub_biome']))

# Calculate non-matches
non_full_matches = all_attempts_set - matched_full_set
non_partial_matches = all_attempts_set - matched_partial_set

# Clean, readable printing for non-matches
def print_non_matches(non_matches, title, n=30):
    print(f"\n=== {title} ===")
    if not non_matches:
        print("All entries matched.\n")
        return
    sample_results = random.sample(list(non_matches), min(n, len(non_matches)))
    for i, (sample_id, sub_biome) in enumerate(sample_results, 1):
        merged = sample_to_merged.get(sample_id, "Merged text not found")
        print(f"{i}. SAMPLE ID: {sample_id}\n   SUB-BIOME: {sub_biome}\n   MERGED TEXT: {merged}\n{'-'*80}")

# Print
print_non_matches(non_full_matches, "NON-FULL MATCHES (no exact full word match)")
print_non_matches(non_partial_matches, "NON-PARTIAL MATCHES (no any word match)")







# Calculate percentages
total_attempts = len(df_sub_biomes_filtered)
percent_full = (len(results_full) / total_attempts) * 100 if total_attempts > 0 else 0
percent_partial = (len(results_partial) / total_attempts) * 100 if total_attempts > 0 else 0

print(f"\nPercentage of FULL matches: {percent_full:.2f}%")
print(f"Percentage of PARTIAL matches: {percent_partial:.2f}%")
