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

# Base directory where metadata files are located
base_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/"

# Optional biome variable
specified_biome = None  # Set to None if you do not want to filter by biome

# Dictionary to keep track of matches
match_count = {}
sample_match_count = {}  # Dictionary to keep track of total matches per sample

# Function to process matches
def process_matches(file_path, sub_biome, sample_id):
    full_matches = set()
    partial_matches = set()
    sub_biome_parts = sub_biome.lower().split()

    sample_match_count_full = 0
    sample_match_count_partial = 0

    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                value_lower = value.lower().strip()

                # Check for full match
                if sub_biome.lower() in value_lower:
                    if key not in full_matches:
                        if key not in match_count:
                            match_count[key] = {'full': 0, 'partial': 0}
                        match_count[key]['full'] += 1
                        full_matches.add(key)
                        sample_match_count_full += 1

                # Check for partial match
                elif any(part in value_lower for part in sub_biome_parts):
                    if key not in partial_matches and key not in full_matches:
                        if key not in match_count:
                            match_count[key] = {'full': 0, 'partial': 0}
                        match_count[key]['partial'] += 1
                        partial_matches.add(key)
                        sample_match_count_partial += 1
                        

    sample_match_count[sample_id] = sample_match_count_full, sample_match_count_partial


# Walk through directories to find relevant metadata files and process them
for sample_id, info in list(gold_dict.items())[:1000]:   # Iterate through a subset or all items
    sub_biome = info[2]
    if specified_biome is None or info[1].lower() == specified_biome.lower():
        subdir = "dir_" + sample_id[-3:]  # Directory based on the last three digits of sample_id
        metadata_filename = f"{sample_id}_clean.txt"
        metadata_filepath = os.path.join(base_dir, subdir, metadata_filename)
    
        if os.path.exists(metadata_filepath):
            process_matches(metadata_filepath, sub_biome, sample_id)

# Convert the match_count dictionary to a DataFrame and sort it
data_items = [(field, counts['full'], counts['partial']) for field, counts in match_count.items()]
match_count_df = pd.DataFrame(data_items, columns=['Field', 'FullMatches', 'PartialMatches'])
match_count_df.sort_values(by='FullMatches', ascending=False, inplace=True)



# Statistical Analysis on field matches
total_matches = match_count_df['FullMatches'] + match_count_df['PartialMatches']
print(f"Total fields: {len(total_matches)}")
print(f"Total mentions: {total_matches.sum()}")
print(f"Mean mentions per field: {total_matches.mean()}")
print(f"Standard deviation: {total_matches.std()}")
print(f"Range of mentions: {total_matches.max() - total_matches.min()}")




# Convert sample_match_count to a DataFrame
sample_match_df = pd.DataFrame.from_dict(sample_match_count, orient='index', columns=['FullMatches', 'PartialMatches'])
sample_match_df['TotalMatches'] = sample_match_df['FullMatches'] + sample_match_df['PartialMatches']

# Statistical Analysis on sample matches
print("\nSample Match Statistics:")
print(f"Total samples: {len(sample_match_df)}")
print(f"Total full matches: {sample_match_df['FullMatches'].sum()}")
print(f"Total partial matches: {sample_match_df['PartialMatches'].sum()}")
print(f"Mean full matches per sample: {sample_match_df['FullMatches'].mean()}")
print(f"Mean partial matches per sample: {sample_match_df['PartialMatches'].mean()}")
print(f"Standard deviation of full matches: {sample_match_df['FullMatches'].std()}")
print(f"Standard deviation of partial matches: {sample_match_df['PartialMatches'].std()}")


# Calculate the average number of fields used per sample
fields_per_sample = [sum(x > 0 for x in matches) for matches in sample_match_df[['FullMatches', 'PartialMatches']].values]
average_fields_per_sample = sum(fields_per_sample) / len(fields_per_sample)
median_mentions = np.median(total_matches)
print(f"Average number of fields mentioning sample origin per sample: {average_fields_per_sample}")
print(f"Median number of fields mentioning sample origin per sample: {median_mentions}")


fields_per_sample = sample_match_df[['FullMatches', 'PartialMatches']].sum(axis=1)
print(f"Average number of fields used per sample: {fields_per_sample.mean()}")
print(f"Median number of fields used per sample: {fields_per_sample.median()}")

print('Top 10 fields:')
print(match_count_df.head(10))









# Plotting
plt.figure(figsize=(12, 8))
bar_width = 0.35  # Width of the bars in the bar plot
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



# Sort the sample_match_df by FullMatches in descending order
sample_match_df = sample_match_df.sort_values(by='FullMatches', ascending=False)
sample_index = range(len(sample_match_df))

# Plotting sample match statistics
plt.figure(figsize=(12, 8))
plt.bar(sample_index, sample_match_df['FullMatches'], bar_width, label='Full matches', color='b')
plt.bar(sample_index, sample_match_df['PartialMatches'], bar_width, bottom=sample_match_df['FullMatches'], label='Partial matches', color='r')

plt.xlabel('Samples')
plt.ylabel('Count of Matches')
plt.title('Total Matches per Sample')
plt.xticks(ticks=sample_index, labels=[], rotation=90, ha="right")  # Remove x-axis labels
plt.legend()
plt.tight_layout()
plt.show()







