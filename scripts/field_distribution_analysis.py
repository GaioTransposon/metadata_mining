#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:30:42 2024

@author: dgaio
"""


import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Files and Paths
# ----------------------------- 
home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")

# -----------------------------
# Ground truth loading & processing
# -----------------------------    
input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

# metadata files 
base_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/"

# optionally filter
specified_biome = None  # options: None, "animal", "water", "plant", "soil", "other"


# keep track of matches
match_count = {}
sample_match_count = {}  

def process_full_matches(file_path, sub_biome, sample_id):
    full_matches = set()
    sample_match_count_full = 0

    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                value_lower = value.lower().strip()

                if sub_biome.lower() in value_lower:
                    if key not in full_matches:
                        if key not in match_count:
                            match_count[key] = {'full': 0, 'partial': 0}  
                        match_count[key]['full'] += 1
                        full_matches.add(key)
                        print(f"Found full match for key: {key}")
                        sample_match_count_full += 1

    print('Full matches:', full_matches)
    current_partial = sample_match_count.get(sample_id, (0, 0))[1]
    sample_match_count[sample_id] = (sample_match_count_full, current_partial)


def process_partial_matches(file_path, sub_biome, sample_id):
    partial_matches = set()
    sub_biome_parts = sub_biome.lower().split()
    sample_match_count_partial = 0

    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                value_lower = value.lower().strip()

                if any(part in value_lower for part in sub_biome_parts):
                    if key not in partial_matches:
                        if key not in match_count:
                            match_count[key] = {'full': 0, 'partial': 0}  
                        match_count[key]['partial'] += 1
                        partial_matches.add(key)
                        print(f"Found partial match for key: {key}")
                        sample_match_count_partial += 1

    print('Partial matches:', partial_matches)
    current_full = sample_match_count.get(sample_id, (0, 0))[0]
    sample_match_count[sample_id] = (current_full, sample_match_count_partial)



for sample_id, info in list(gold_dict.items())[:1000]:
    sub_biome = info[2]
    if not specified_biome or info[1].lower() == specified_biome.lower():
        subdir = "dir_" + sample_id[-3:]
        metadata_filename = f"{sample_id}_clean.txt"
        metadata_filepath = os.path.join(base_dir, subdir, metadata_filename)

        if os.path.exists(metadata_filepath):
            print('\n#####')
            print(sample_id)
            print(sub_biome)
            process_full_matches(metadata_filepath, sub_biome, sample_id)
            print('#')
            process_partial_matches(metadata_filepath, sub_biome, sample_id)

        

data_items = [(field, counts['full'], counts['partial']) for field, counts in match_count.items()]
match_count_df = pd.DataFrame(data_items, columns=['Field', 'FullMatches', 'PartialMatches'])
match_count_df.sort_values(by='FullMatches', ascending=False, inplace=True)
#print(match_count_df)

# convert to a df
sample_match_df = pd.DataFrame.from_dict(sample_match_count, orient='index', columns=['FullMatches', 'PartialMatches'])
sample_match_df['TotalMatches'] = sample_match_df['FullMatches'] + sample_match_df['PartialMatches']


# in how many distinct fields, for 1000 samples, was the sample origin reported? 
fields_full_match = {key for key, value in match_count.items() if value['full'] > 0}
fields_partial_match = {key for key, value in match_count.items() if value['partial'] > 0}

# which fields are the most popularly used to report sample origin (full matches)? 
most_popular_full = match_count_df.nlargest(10, 'FullMatches')[['Field', 'FullMatches']]

# which fields are the most popularly used to report sample origin (partial matches)? 
most_popular_partial = match_count_df.nlargest(5, 'PartialMatches')[['Field', 'PartialMatches']]

# in how many fields can one find the full sample origin, per sample? (mean, median and sd)
mean_full = sample_match_df['FullMatches'].mean()
median_full = sample_match_df['FullMatches'].median()
std_full = sample_match_df['FullMatches'].std()
# in how many fields can one find at least part of the sample origin info, per sample? (mean, median and sd)
mean_partial = sample_match_df['PartialMatches'].mean()
median_partial = sample_match_df['PartialMatches'].median()
std_partial = sample_match_df['PartialMatches'].std()

print('#################')
print('In how many distinct fields, for 1000 samples, was the sample origin reported?')
print(f"For 1000 samples, the sample origin can be found in {len(fields_full_match)} distinct fields.")
print(f"For 1000 samples, at least part of the sample origin can be found in {len(fields_partial_match)} distinct fields.")
print('##')
print("Most popular fields for full matches:")
print(most_popular_full)
print('##')
print("Most popular fields for partial matches:")
print(most_popular_partial)
print('##')
print('in how many fields can one find the full sample origin, per sample?')
print(f"Mean full matches per sample: {mean_full}")
print(f"Median full matches per sample: {median_full}")
print(f"Standard deviation of full matches per sample: {std_full}")
print('##')
print('in how many fields can one find the partial sample origin, per sample?')
print(f"Mean partial matches per sample: {mean_partial}")
print(f"Median partial matches per sample: {median_partial}")
print(f"Standard deviation of partial matches per sample: {std_partial}")
print('#################')


##############
# count words in each metadata field
def count_words_in_fields(base_dir, gold_dict, top_fields):
    word_counts = {field: [] for field in top_fields}  

    for sample_id, info in list(gold_dict.items()):
        subdir = "dir_" + sample_id[-3:]
        metadata_filename = f"{sample_id}_clean.txt"
        metadata_filepath = os.path.join(base_dir, subdir, metadata_filename)
        
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, 'r') as file:
                for line in file:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        if key in top_fields:
                            word_counts[key].append(len(value.split()))

    return word_counts

top_fields = list(most_popular_full['Field'])[0:5]
word_counts = count_words_in_fields(base_dir, gold_dict, top_fields)

for field, counts in word_counts.items():
    if counts:
        print(f"Average word count for {field}: {sum(counts) / len(counts)}")
##############


# Plot
plt.figure(figsize=(12, 8))
bar_width = 0.35  
index = range(len(match_count_df))

plt.bar(index, match_count_df['FullMatches'], bar_width, label='Full matches', color='b')
plt.bar(index, match_count_df['PartialMatches'], bar_width, bottom=match_count_df['FullMatches'], label='Partial matches', color='r')

title_biome = specified_biome if specified_biome else "all biomes"
plt.xlabel('Metadata Fields')
plt.ylabel('Count of Matches')
plt.title(f'Frequency of Metadata Fields Matching Sub-biome (Metadata Informative Fields of Sample Origin) - Biome: "{title_biome}"')
plt.xticks(ticks=index, labels=match_count_df['Field'], rotation=90, ha="right")
plt.legend()
plt.tight_layout()
plt.show()




# # save to csv:
# most_popular_full.to_csv('most_popular_full_matches.csv', index=False)
# most_popular_partial.to_csv('most_popular_partial_matches.csv', index=False)





