#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:26:01 2025

@author: danielagaio
"""



import pandas as pd
import re
from tqdm import tqdm
import random

# -------------------------
# 1. Load GPT_biomes.txt and filter to 'animal' or 'plant'
# -------------------------
gpt_biomes_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt'
mapping_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/sample_taxid_mapping_clean_fixed_matlas2024.tsv'
gpt_sub_biomes_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_sub_biomes.txt'

df_biomes = pd.read_csv(gpt_biomes_fp, sep='\t', header=None, names=['sample_id', 'biome'])
df_biomes_filtered = df_biomes[df_biomes['biome'].isin(['animal', 'plant'])]
sample_ids_biomes = df_biomes_filtered['sample_id'].unique()

# -------------------------
# 2. Load GPT_sub_biomes.txt and filter by sample_ids from step 1
# -------------------------
df_sub_biomes = pd.read_csv(gpt_sub_biomes_fp, sep='\t', header=None, names=['sample_id', 'sub_biome'])
df_sub_biomes_filtered = df_sub_biomes[df_sub_biomes['sample_id'].isin(sample_ids_biomes)].copy()

# -------------------------
# 3. Load Taxonomic mappings, filter, merge, clean
# -------------------------
df_mapping = pd.read_csv(mapping_fp, sep='\t')
df_mapping_filtered = df_mapping[df_mapping.iloc[:, 0].isin(sample_ids_biomes)].copy()

# Merge taxonomic mapping columns into 'merged'
df_mapping_filtered['merged'] = df_mapping_filtered.iloc[:, 1:].astype(str).apply(lambda row: ' '.join(row), axis=1)

# Track before cleaning
total_before_nan_removal = len(df_mapping_filtered)

# Remove rows where 'merged' is only 'nan nan nan...'
df_mapping_filtered['merged_clean'] = df_mapping_filtered['merged'].apply(lambda x: x.replace('nan', '').strip())
nan_only_sample_ids = df_mapping_filtered[df_mapping_filtered['merged_clean'] == ''][df_mapping_filtered.columns[0]].tolist()
df_mapping_filtered = df_mapping_filtered[df_mapping_filtered['merged_clean'] != ''].copy()
df_mapping_filtered['merged'] = df_mapping_filtered['merged_clean']
df_mapping_filtered.drop(columns=['merged_clean'], inplace=True)

total_after_nan_removal = len(df_mapping_filtered)
removed_nans = total_before_nan_removal - total_after_nan_removal

# -------------------------
# 4. Sync both df_sub_biomes_filtered and df_mapping_filtered to only common sample_ids
# -------------------------
common_sample_ids = set(df_sub_biomes_filtered['sample_id']).intersection(set(df_mapping_filtered.iloc[:, 0]))
df_sub_biomes_filtered = df_sub_biomes_filtered[df_sub_biomes_filtered['sample_id'].isin(common_sample_ids)].copy()
df_mapping_filtered = df_mapping_filtered[df_mapping_filtered.iloc[:, 0].isin(common_sample_ids)].copy()

# -------------------------
# 5. Remove 'species' from merged text
# -------------------------
df_mapping_filtered['merged'] = df_mapping_filtered['merged'].str.replace('species', '', case=False, regex=False)
df_sub_biomes_filtered['sub_biome'] = df_sub_biomes_filtered['sub_biome'].str.replace('species', '', case=False, regex=False)

# Create dict of sample_id -> merged_text
sample_to_merged = df_mapping_filtered.set_index(df_mapping_filtered.columns[0])['merged'].astype(str).to_dict()

# -------------------------
# 6. Matching logic
# -------------------------
results_full = []
results_partial = []

for _, row in tqdm(df_sub_biomes_filtered.iterrows(), total=len(df_sub_biomes_filtered), desc="Sample-wise matching"):
    sample_id = row['sample_id']
    sub_biome = row['sub_biome']
    
    if not isinstance(sub_biome, str) or sample_id not in sample_to_merged:
        continue
    
    merged_text = sample_to_merged[sample_id]
    
    # Full match (whole phrase with word boundaries)
    pattern_full = r'\b' + re.escape(sub_biome.strip()) + r'\b'
    if re.search(pattern_full, merged_text):
        results_full.append((sample_id, sub_biome, merged_text))
        continue
    
    # Partial match (any word as whole word)
    if any(re.search(r'\b' + re.escape(word) + r'\b', merged_text, flags=re.IGNORECASE) for word in sub_biome.strip().split()):
        results_partial.append((sample_id, sub_biome, merged_text))

# Deduplicate
results_full = list(set(results_full))
results_partial = list(set(results_partial))

# -------------------------
# Reporting functions
# -------------------------
def print_readable_samples(results, title, n=30):
    print(f"\n=== {title} ===")
    if not results:
        print("No matches found.\n")
        return
    sample_results = random.sample(results, min(n, len(results)))
    for i, (sample_id, sub_biome, merged) in enumerate(sample_results, 1):
        print(f"{i}. SAMPLE ID: {sample_id}\n   SUB-BIOME: {sub_biome}\n   MERGED TEXT: {merged}\n{'-'*80}")

def print_non_matches(non_matches, title, n=30):
    print(f"\n=== {title} ===")
    if not non_matches:
        print("All entries matched.\n")
        return
    sample_results = random.sample(list(non_matches), min(n, len(non_matches)))
    for i, (sample_id, sub_biome) in enumerate(sample_results, 1):
        merged = sample_to_merged.get(sample_id, "Merged text not found")
        print(f"{i}. SAMPLE ID: {sample_id}\n   SUB-BIOME: {sub_biome}\n   MERGED TEXT: {merged}\n{'-'*80}")

# -------------------------
# Reporting matches and non-matches
# -------------------------
print_readable_samples(results_full, "FULL MATCHES (whole phrase match)")
print_readable_samples(results_partial, "PARTIAL MATCHES (any word as whole word match)")

matched_full_set = set((sample_id, sub_biome) for sample_id, sub_biome, _ in results_full)
matched_partial_set = set((sample_id, sub_biome) for sample_id, sub_biome, _ in results_partial)
all_attempts_set = set(zip(df_sub_biomes_filtered['sample_id'], df_sub_biomes_filtered['sub_biome']))

non_full_matches = all_attempts_set - matched_full_set
non_partial_matches = all_attempts_set - matched_partial_set

print_non_matches(non_full_matches, "NON-FULL MATCHES (no exact full phrase match)")
print_non_matches(non_partial_matches, "NON-PARTIAL MATCHES (no whole word match)")

# -------------------------
# Summary statistics
# -------------------------
total_attempts = len(df_sub_biomes_filtered)
percent_full = (len(results_full) / total_attempts) * 100 if total_attempts > 0 else 0
percent_partial = (len(results_partial) / total_attempts) * 100 if total_attempts > 0 else 0
percent_removed_nans = (removed_nans / total_before_nan_removal) * 100 if total_before_nan_removal > 0 else 0

print(f"\nTotal samples in taxonomic mappings (after GPT biome filtering): {total_before_nan_removal}")
print(f"Samples with only 'nan' merged text (removed early): {removed_nans} ({percent_removed_nans:.2f}%)")
print(f"Samples remaining after 'nan' removal and syncing: {len(df_mapping_filtered)}")

print(f"\nTotal samples attempted for matching: {total_attempts}")
print(f"Percentage of FULL matches: {percent_full:.2f}%")
print(f"Percentage of PARTIAL matches: {percent_partial:.2f}%")



# -------------------------
# Save clean prepared dataframe for GPT API matching later
# -------------------------
# Merge df_sub_biomes_filtered and df_mapping_filtered on 'sample_id'
df_final = pd.merge(
    df_sub_biomes_filtered[['sample_id', 'sub_biome']],
    df_mapping_filtered[[df_mapping_filtered.columns[0], 'merged']],
    left_on='sample_id',
    right_on=df_mapping_filtered.columns[0]
)

# Remove possible duplicate column
df_final = df_final[['sample_id', 'sub_biome', 'merged']]

# Save to CSV (choose your preferred path)
df_final.to_csv('/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/gpt_matching_ready_dataset.csv', index=False)

print(f"\nSaved {len(df_final)} cleaned samples ready for GPT matching to 'gpt_matching_ready_dataset.csv'.")



