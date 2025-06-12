#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:30:42 2024

@author: dgaio
"""


# runs as: 
# python ~/github/metadata_mining/scripts/field_distrib_analysis.py --split_dirs '~/MicrobeAtlasProject/sample_info_split_dirs/'



import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# -----------------------------
# Files and Paths
# ----------------------------- 
#home_dir = os.getenv('HOME')
#work_dir = os.path.join(home_dir, "/MicrobeAtlasProject")  
work_dir = os.path.expanduser("~/MicrobeAtlasProject/")


# -----------------------------
# Ground truth loading & processing
# -----------------------------    
#input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
input_gold_dict = os.path.expanduser("github/metadata_mining/source_data/gold_dict.pkl")


with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

# metadata files 
parser = argparse.ArgumentParser(description="Analyze metadata size reductions")
parser.add_argument("--split_dirs", required=True, help="Path to directory containing metadata files split across subdirectories ")
args = parser.parse_args()


base_dir = os.path.expanduser(args.split_dirs)



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

# Define all biomes to process
biomes = ["all", "animal", "water", "plant", "soil", "other"]
all_match_counts = {}  # Dictionary to store match counts for each biome

for specified_biome in biomes:
    # Reset match counts for each biome
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
                            sample_match_count_full += 1

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
                            sample_match_count_partial += 1

        current_full = sample_match_count.get(sample_id, (0, 0))[0]
        sample_match_count[sample_id] = (current_full, sample_match_count_partial)

    # Process samples for current biome
    for sample_id, info in list(gold_dict.items())[:1000]:
        sub_biome = info[2]
        current_biome = info[1].lower()
        
        # Skip if specified_biome is not "all" and doesn't match the current sample's biome
        if specified_biome != "all" and current_biome != specified_biome.lower():
            continue
            
        subdir = "dir_" + sample_id[-3:]
        metadata_filename = f"{sample_id}_clean.txt"
        metadata_filepath = os.path.join(base_dir, subdir, metadata_filename)

        if os.path.exists(metadata_filepath):
            process_full_matches(metadata_filepath, sub_biome, sample_id)
            process_partial_matches(metadata_filepath, sub_biome, sample_id)

    # Create dataframe from match counts for this biome
    data_items = [(field, counts['full'], counts['partial']) for field, counts in match_count.items()]
    match_count_df = pd.DataFrame(data_items, columns=['Field', 'FullMatches', 'PartialMatches'])
    match_count_df.sort_values(by='FullMatches', ascending=False, inplace=True)
    
    # Store dataframe for this biome
    all_match_counts[specified_biome] = match_count_df

# First, filter the 'all' biomes dataframe to get the common set of fields
all_biomes_df = all_match_counts['all']
filtered_all = all_biomes_df[all_biomes_df['PartialMatches'] > 5]
common_fields = list(filtered_all['Field'])

# Create the figure with 6 subplots (3 rows, 2 columns)
# Make the figure more elongated by increasing the height
fig, axes = plt.subplots(3, 2, figsize=(20, 30))  # Changed height from 24 to 30
axes = axes.flatten()  # Flatten for easier indexing

# Plot for each biome
for i, biome in enumerate(biomes):
    ax = axes[i]
    df = all_match_counts[biome]
    
    # Create a consistent dataframe with all common fields, filling with zeros for missing fields
    consistent_data = []
    
    for field in common_fields:
        field_row = df[df['Field'] == field]
        if len(field_row) > 0:
            full_matches = field_row['FullMatches'].values[0]
            partial_matches = field_row['PartialMatches'].values[0]
        else:
            full_matches = 0
            partial_matches = 0
        
        consistent_data.append((field, full_matches, partial_matches))
    
    consistent_df = pd.DataFrame(consistent_data, columns=['Field', 'FullMatches', 'PartialMatches'])
    
    # Plot bars
    bar_width = 0.8
    index = np.arange(len(consistent_df))
    
    # Plot full matches and partial matches as stacked bars
    p1 = ax.bar(index, consistent_df['FullMatches'], bar_width, label='Full matches', color='b')
    p2 = ax.bar(index, consistent_df['PartialMatches'], bar_width, 
               bottom=consistent_df['FullMatches'], label='Partial matches', color='r')
    
    # Set titles and labels
    ax.set_title(f"Biome: {biome}", fontsize=14)
    
    # Only set x-axis labels for bottom plots (last row)
    if i >= 4:  # Bottom row
        ax.set_xlabel('Metadata fields', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(consistent_df['Field'], rotation=90, ha='center', fontsize=10)
    else:
        ax.set_xticks(index)
        ax.set_xticklabels([])
        
    # Only set y-axis label for left plots in the MIDDLE row (position 2)
    if i == 2:  # Middle row, left column
        ax.set_ylabel('Match count', fontsize=14)  # Changed text and increased font size
    
    # Set legend only for the first plot
    if i == 0:
        ax.legend(fontsize=10)
    
    # Make y-axis flexible for each plot
    total_matches = consistent_df['FullMatches'] + consistent_df['PartialMatches']
    if total_matches.max() > 0:
        ax.set_ylim(0, total_matches.max() * 1.1)  # Add 10% padding
    else:
        ax.set_ylim(0, 10)  # Default limit if no matches

# Adjust layout and add more space at the top for the main title
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, top=0.95)  # Added top parameter to create more space


# Save figure
my_png = os.path.join(work_dir, "metadata_fields_by_biome_test.png")  
plt.savefig(my_png, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# save to csv:
mostpopfull = os.path.join(work_dir, "most_popular_full_matches.csv")  
most_popular_full.to_csv(mostpopfull, index=False)
mostpoppart = os.path.join(work_dir, "most_popular_partial_matches.csv")  
most_popular_full.to_csv(mostpoppart, index=False)







