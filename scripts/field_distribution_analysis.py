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

def process_full_matches(file_path, sub_biome, sample_id):
    full_matches = set()
    sample_match_count_full = 0

    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                value_lower = value.lower().strip()

                # Check for full match
                if sub_biome.lower() in value_lower:
                    if key not in full_matches:
                        if key not in match_count:
                            match_count[key] = {'full': 0, 'partial': 0}  # Initialize both counts
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

                # Check for partial match
                if any(part in value_lower for part in sub_biome_parts):
                    if key not in partial_matches:
                        if key not in match_count:
                            match_count[key] = {'full': 0, 'partial': 0}  # Ensure proper initialization
                        match_count[key]['partial'] += 1
                        partial_matches.add(key)
                        print(f"Found partial match for key: {key}")
                        sample_match_count_partial += 1

    print('Partial matches:', partial_matches)
    current_full = sample_match_count.get(sample_id, (0, 0))[0]
    sample_match_count[sample_id] = (current_full, sample_match_count_partial)




# Assuming 'gold_dict' and other setup from your provided code
for sample_id, info in list(gold_dict.items())[:3]:
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

            





# Assuming match_count is updated correctly by the modified functions
# Convert the match_count dictionary to a DataFrame and sort it
data_items = [(field, counts['full'], counts['partial']) for field, counts in match_count.items()]
match_count_df = pd.DataFrame(data_items, columns=['Field', 'FullMatches', 'PartialMatches'])

# Sort the DataFrame by 'FullMatches' in descending order
match_count_df.sort_values(by='FullMatches', ascending=False, inplace=True)

# Display the DataFrame to see the sorted values
print(match_count_df)






# in how many fields can one find the full sample origin, per sample? (mean, median and sd)
# in how many fields can one find at least part of the sample origin info, per sample? (mean, median and sd)





# in how many distinct fields does one have to look through to find sample origin? 
# "for 1000 samples, the sample origin cabn be found in  ... to ... fields" 


# tot count of distinct fields containing all sample origin (full match)

# how many fields does one have to look through to find at least part of info about sample origin? 
# tot count of distinct fields containing all sample origin (partial match)

# which fields are the most popularly used to report sample origin (full matches)? 
# are these fields the same when looking at partial matches? 







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







