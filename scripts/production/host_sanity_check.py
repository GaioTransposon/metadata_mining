#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:21:07 2025

@author: danielagaio
"""



# -------------------------------------------
# Libraries & Setup
# -------------------------------------------
import pandas as pd
import re
import random
import string
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import inflect
from rapidfuzz import fuzz
from functools import lru_cache
import nltk
import json
import os
from collections import Counter

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
p = inflect.engine()

# -------------------------------------------
# Helper Functions
# -------------------------------------------
def clean_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    return text.translate(str.maketrans('', '', string.punctuation)).lower().strip()

@lru_cache(maxsize=None)
def lemmatize_and_singularize_cached(text):
    text = clean_text(text)
    return ' '.join([p.singular_noun(lemmatizer.lemmatize(word)) or word for word in text.split()])

@lru_cache(maxsize=None)
def pluralize_text_cached(text):
    text = clean_text(text)
    return ' '.join([p.plural(word) for word in text.split()])

def print_readable_samples(results, title, n=30):
    print(f"\n=== {title} ===")
    if not results:
        print("No matches found.\n")
        return
    for i, (sample_id, sub_biome, merged) in enumerate(random.sample(results, min(n, len(results))), 1):
        print(f"{i}. SAMPLE ID: {sample_id}\n   SUB-BIOME: {sub_biome}\n   MERGED TEXT: {merged}\n{'-'*80}")

def print_non_matches(non_matches, title, n=30):
    print(f"\n=== {title} ===")
    if not non_matches:
        print("All entries matched.\n")
        return
    for i, (sample_id, sub_biome) in enumerate(random.sample(list(non_matches), min(n, len(non_matches))), 1):
        merged = sample_to_merged.get(sample_id, "Merged text not found")
        print(f"{i}. SAMPLE ID: {sample_id}\n   SUB-BIOME: {sub_biome}\n   MERGED TEXT: {merged}\n{'-'*80}")

# -------------------------------------------
# 1. Load and Filter Data
# -------------------------------------------
gpt_biomes_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt'
mapping_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/sample_taxid_mapping_clean_fixed_matlas2024.tsv'
gpt_sub_biomes_fp = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_sub_biomes.txt'

df_biomes = pd.read_csv(gpt_biomes_fp, sep='\t', header=None, names=['sample_id', 'biome'])
df_biomes_filtered = df_biomes[df_biomes['biome'].isin(['animal', 'plant'])]
sample_ids_biomes = df_biomes_filtered['sample_id'].unique()

df_sub_biomes = pd.read_csv(gpt_sub_biomes_fp, sep='\t', header=None, names=['sample_id', 'sub_biome'])
df_sub_biomes_filtered = df_sub_biomes[df_sub_biomes['sample_id'].isin(sample_ids_biomes)].copy()

df_mapping = pd.read_csv(mapping_fp, sep='\t')
df_mapping_filtered = df_mapping[df_mapping.iloc[:, 0].isin(sample_ids_biomes)].copy()
df_mapping_filtered['merged'] = df_mapping_filtered.iloc[:, 1:].astype(str).apply(lambda row: ' '.join(row), axis=1)
df_mapping_filtered['merged'] = df_mapping_filtered['merged'].apply(lambda x: x.replace('nan', '').strip())
df_mapping_filtered = df_mapping_filtered[df_mapping_filtered['merged'] != '']

# Sync to common sample_ids
common_sample_ids = set(df_sub_biomes_filtered['sample_id']).intersection(df_mapping_filtered.iloc[:, 0])
df_sub_biomes_filtered = df_sub_biomes_filtered[df_sub_biomes_filtered['sample_id'].isin(common_sample_ids)]
df_mapping_filtered = df_mapping_filtered[df_mapping_filtered.iloc[:, 0].isin(common_sample_ids)]

# Clean up 'species'
df_mapping_filtered['merged'] = df_mapping_filtered['merged'].str.replace('species', '', case=False, regex=False)
df_sub_biomes_filtered['sub_biome'] = df_sub_biomes_filtered['sub_biome'].str.replace('species', '', case=False, regex=False)

sample_to_merged = df_mapping_filtered.set_index(df_mapping_filtered.columns[0])['merged'].astype(str).to_dict()

# -------------------------------------------
# 2. Matching Logic (Full & Partial)
# -------------------------------------------
results_full, results_partial = [], []

for _, row in tqdm(df_sub_biomes_filtered.iterrows(), total=len(df_sub_biomes_filtered), desc="Sample-wise matching (improved)"):
    sample_id = row['sample_id']
    sub_biome_raw = row['sub_biome']
    merged_text = sample_to_merged.get(sample_id, "")

    if not isinstance(sub_biome_raw, str) or not merged_text:
        continue

    sub_biome_clean = clean_text(sub_biome_raw.replace('-', ' '))
    merged_clean = clean_text(merged_text)

    if re.search(rf'\b{re.escape(sub_biome_clean)}\b', merged_clean):
        results_full.append((sample_id, sub_biome_raw, merged_text))
    elif any(re.search(rf'\b{re.escape(word)}\b', merged_clean) for word in sub_biome_clean.split()):
        results_partial.append((sample_id, sub_biome_raw, merged_text))

results_full = list(set(results_full))
results_partial = list(set(results_partial))

# -------------------------------------------
# 3. Fuzzy & Normalized Matching for Non-Partial Matches
# -------------------------------------------
matched_partial_set = set((s, sb) for s, sb, _ in results_partial)
all_attempts_set = set(zip(df_sub_biomes_filtered['sample_id'], df_sub_biomes_filtered['sub_biome']))
non_partial_matches = all_attempts_set - matched_partial_set

df_non_partial = pd.DataFrame(list(non_partial_matches), columns=['sample_id', 'sub_biome'])
df_non_partial['merged'] = df_non_partial['sample_id'].map(sample_to_merged)
df_non_partial['sub_biome_singular'] = df_non_partial['sub_biome'].map(lemmatize_and_singularize_cached)
df_non_partial['sub_biome_plural'] = df_non_partial['sub_biome'].map(pluralize_text_cached)
df_non_partial['merged_norm'] = df_non_partial['merged'].map(lemmatize_and_singularize_cached)

results_norm_partial, results_norm_fuzzy = [], []

for _, row in tqdm(df_non_partial.iterrows(), total=len(df_non_partial), desc="Partial+Fuzzy matching"):
    sample_id, merged_norm = row['sample_id'], row['merged_norm']
    forms_to_try = [row['sub_biome_singular'], row['sub_biome_plural']]

    if any(re.search(rf'\b{re.escape(word)}\b', merged_norm) for form in forms_to_try if form for word in form.split()):
        results_norm_partial.append((sample_id, row['sub_biome'], row['merged']))
    elif fuzz.token_set_ratio(row['sub_biome_singular'], merged_norm) >= 85:
        results_norm_fuzzy.append((sample_id, row['sub_biome'], row['merged'], fuzz.token_set_ratio(row['sub_biome_singular'], merged_norm)))

rescued_norm_partial_set = set((s, sb) for s, sb, _ in results_norm_partial)
rescued_norm_fuzzy_set = set((s, sb) for s, sb, _, _ in results_norm_fuzzy)
rescued_all_set = rescued_norm_partial_set.union(rescued_norm_fuzzy_set)

still_unmatched_after_all = non_partial_matches - rescued_all_set

# -------------------------------------------
# 4. Reporting Summary
# -------------------------------------------
total_samples = len(df_sub_biomes_filtered)
full_matches = len(results_full)
partial_matches = len(results_partial)
rescued_total = len(rescued_all_set)
unmatched_total = len(still_unmatched_after_all)

print(f"\n=== SUMMARY REPORT ===")
print(f"Total samples attempted: {total_samples}")
print(f"Full matches (whole phrase): {full_matches} ({(full_matches / total_samples) * 100:.2f}%)")
print(f"Partial matches (any word): {partial_matches} ({(partial_matches / total_samples) * 100:.2f}%)")
print(f"Rescued from unmatched partial (normalized/fuzzy): {rescued_total} ({(rescued_total / len(non_partial_matches) * 100):.2f}%)")
print(f"Remaining unmatched after all steps: {unmatched_total} ({(unmatched_total / total_samples) * 100:.2f}%)")

# print_readable_samples(results_full, "FULL MATCHES")
# print_readable_samples(results_partial, "PARTIAL MATCHES")
# print_readable_samples(results_norm_partial, "RESCUED PARTIAL (normalized matching)")
# print_readable_samples(results_norm_fuzzy, "RESCUED FUZZY (score >=85)")
# print_non_matches(still_unmatched_after_all, "STILL UNMATCHED AFTER ALL STEPS")




# -------------------------------------------
# 5. Persistent Comment Game on unmatched samples
# -------------------------------------------

COMMENT_FILE = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/comment_dict.json'

def load_comment_dict():
    if os.path.exists(COMMENT_FILE):
        with open(COMMENT_FILE, 'r') as f:
            return json.load(f)
    else:
        print("\nNo previous comments found. Starting fresh.")
        return {}

def save_comment_dict(comment_dict):
    with open(COMMENT_FILE, 'w') as f:
        json.dump(comment_dict, f, indent=2)

# Load existing comments
comment_dict = load_comment_dict()
print(f"\nLoaded {len(comment_dict)} previously commented samples.")

# Make the list stable and reproducible
random.seed(42)
unmatched_list = list(still_unmatched_after_all)
random.shuffle(unmatched_list)

print(f"\n=== COMMENT GAME START ===")
print(f"Total unmatched samples to review: {len(unmatched_list)}")
print(f"Type 'q' or 'exit' at any time to quit.\n")

for sample_id, sub_biome in unmatched_list:
    if sample_id in comment_dict:
        continue

    merged = sample_to_merged.get(sample_id, "Merged text not found")

    print(f"\nSample ID: {sample_id}")
    print(f"Sub-Biome: {sub_biome}")
    print(f"Taxonomic Assignment: {merged}")
    print("-" * 80)

    user_input = input("\nYour comment (or 'q' to quit): ").strip()
    if user_input.lower() in ['q', 'exit']:
        print("\nExiting comment game...")
        break

    comment_dict[sample_id] = {
        'sub_biome': sub_biome,
        'taxonomic_assign': merged,
        'my_comment': user_input
    }

    save_comment_dict(comment_dict)
    print(f"Comment saved. Total comments so far: {len(comment_dict)}")

print(f"\n=== COMMENT GAME ENDED ===")
print(f"Total commented samples: {len(comment_dict)} (saved to '{COMMENT_FILE}')")

# -------------------------------------------
# 6. Comment Dictionary Statistics
# -------------------------------------------

def show_comment_stats(comment_dict):
    print(f"\n=== COMMENT DICTIONARY STATS ===")
    total_comments = len(comment_dict)
    print(f"Total commented samples: {total_comments}")

    all_comments = [entry['my_comment'] for entry in comment_dict.values()]
    comment_counts = Counter(all_comments)

    print("\nComment distribution:")
    for comment, count in comment_counts.most_common():
        percent = (count / total_comments) * 100
        print(f"- {comment}: {count} samples ({percent:.2f}%)")

show_comment_stats(comment_dict)

# -------------------------------------------
# 7. Filter Samples by Comment Category (example: 'nh')
# -------------------------------------------

# nh: GPT-subbiome is lacking host
# w: wording; taxonomic assignment doesn’t match with sub-biome although they mean the same
# nm: no match 

def filter_by_comment_keyword(comment_dict, keyword):
    filtered = {sample_id: info for sample_id, info in comment_dict.items() if keyword.lower() in info['my_comment'].lower()}
    print(f"\n=== SAMPLES WITH '{keyword}' IN COMMENTS ===")
    print(f"Total: {len(filtered)} samples\n")

    for sample_id, info in filtered.items():
        print(f"Sample ID: {sample_id}")
        print(f"Sub-Biome: {info['sub_biome']}")
        print(f"Taxonomic Assignment: {info['taxonomic_assign']}")
        print(f"My Comment: {info['my_comment']}")
        print('-' * 80)

# Example call to filter 'nh'
filter_by_comment_keyword(comment_dict, 'nh')

# -------------------------------------------
# 8. Correct Comment for a Given Sample ID
# -------------------------------------------

def correct_comment(comment_dict):
    sample_id_to_edit = input("\nEnter the SAMPLE ID you want to correct (or 'q' to quit): ").strip()

    if sample_id_to_edit.lower() in ['q', 'exit']:
        print("Exited.")
        return

    if sample_id_to_edit not in comment_dict:
        print(f"Sample ID '{sample_id_to_edit}' not found in the comments dictionary.")
        return

    entry = comment_dict[sample_id_to_edit]
    print(f"\nCurrent info for {sample_id_to_edit}:")
    print(f"Sub-Biome: {entry['sub_biome']}")
    print(f"Taxonomic Assignment: {entry['taxonomic_assign']}")
    print(f"Current Comment: {entry['my_comment']}")

    new_comment = input("\nEnter your NEW comment: ").strip()
    confirm = input(f"Are you sure you want to replace '{entry['my_comment']}' with '{new_comment}'? (y/n): ").strip().lower()

    if confirm == 'y':
        comment_dict[sample_id_to_edit]['my_comment'] = new_comment
        save_comment_dict(comment_dict)
        print(f"\nComment updated for {sample_id_to_edit}.")
    else:
        print("Cancelled. No changes made.")

# Example usage
# correct_comment(comment_dict)
