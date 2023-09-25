#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:55:43 2023

@author: dgaio
"""



GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.json"
CSV_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv"


# =============================================================================
# import pandas as pd
# import json
# 
# def load_gold_dict(filename):
#     try:
#         with open(filename, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {}
# 
# def save_gold_dict(gold_dict, filename):
#     with open(filename, 'w') as f:
#         json.dump(gold_dict, f)
# 
# 
# def get_sub_df_with_lowest_count(df, gold_dict, iteration):
#     min_count = float('inf')
#     min_biome = None
#     for biome in df['biome'].unique():
#         if biome == 'unknown' and iteration % 10 != 0:  # Skip 'unknown' most of the time (this is because most unknowns turn out to be animal)
#             continue
#         if biome not in gold_dict:
#             gold_dict[biome] = []
#         count = len(gold_dict[biome])
#         if count < min_count:
#             min_count = count
#             min_biome = biome
#     sub_df = df[df['biome'] == min_biome].sort_values(by='biome_counts', ascending=False)
#     return sub_df
# 
# 
# def print_gold_dict_stats(gold_dict):
#     print("Gold dictionary statistics:")
#     for biome, pmids in gold_dict.items():
#         print(f"{biome}: {len(pmids)}")
#         
# 
# def main():
#     filename = CSV_PATH
#     df = pd.read_csv(filename)
#     
#     # Group by 'pmid' and 'biome' and count the rows for each group
#     grouped_df = df.groupby(['pmid', 'biome']).size().reset_index(name='biome_counts')
# 
#     # Optionally, merge this information back into the original DataFrame, if needed.
#     merged_df = pd.merge(df[['pmid', 'title', 'abstract']].drop_duplicates(), grouped_df, on='pmid')
#     
#     # Replace 'aquatic' with 'water' in the 'biome' column
#     merged_df['biome'] = merged_df['biome'].replace('aquatic', 'water')
# 
#     # Sort by title and (evt) take the first n rows
#     df = merged_df.sort_values(by='title')         #.head(20)
#     
#     gold_dict_filename = GOLD_DICT_PATH
#     gold_dict = load_gold_dict(gold_dict_filename)
#     
#     if 'laboratory' not in gold_dict:
#         gold_dict['laboratory'] = []
#     
#     # Remove PMIDs already in gold_dict from DataFrame
#     already_processed_pmids = [pmid for pmid_list in gold_dict.values() for pmid in pmid_list]
#     df = df[~df['pmid'].isin(already_processed_pmids)]
# 
#     iteration = 0  # Initialize an iteration counter
#     while True:
#         iteration += 1  # Increment the iteration counter
#         sub_df = get_sub_df_with_lowest_count(df, gold_dict, iteration)
#         
#         if sub_df.empty:
#             print("No more records to process for the least populated biome. Exiting.")
#             print("Current gold_dict content:")
#             print(gold_dict)
#             print_gold_dict_stats(gold_dict)
#             break
# 
#         row = sub_df.iloc[0]
# 
#         question = f"\n\n\nIs the biome: {row['biome']}?"
#         print(f"\n\n\n\n\n\nTitle: {row['title']}")
#         print(f"\nAbstract: {row['abstract']}\n\n")
#         answer = input(question + " (y/n/q): ")
# 
#         if answer == 'q':
#             print("Current gold_dict content:")
#             print(gold_dict)
#             print_gold_dict_stats(gold_dict)
#             break
#         elif answer == 'y':
#             gold_dict[row['biome']].append(row['pmid'])
#             df = df[df['pmid'] != row['pmid']]
#         else:
#             new_biome = input("Which biome is it? (a for animal, w for water, s for soil, p for plant, l for laboratory, u for unknown): ")
#             biome_map = {'a': 'animal', 'w': 'water', 's': 'soil', 'p': 'plant', 'l': 'laboratory', 'u': 'unknown'} 
#             new_biome_str = biome_map[new_biome]
#             if new_biome_str not in gold_dict:
#                 gold_dict[new_biome_str] = []
#             gold_dict[new_biome_str].append(row['pmid'])
#             df = df[df['pmid'] != row['pmid']]
# 
#     save_gold_dict(gold_dict, gold_dict_filename)
# 
# 
# if __name__ == "__main__":
#     main()
# =============================================================================



path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"


import pandas as pd
import os
import pickle

# # Sample dataframe
# df = pd.DataFrame({
#     'sample': ['DRS001007', 'DRS001008', 'DRS001009', 'DRS001010'],
#     'biome': ['water', 'water', 'animal', 'soil'],
#     'pmid': [987654, 10111213, 987654, 14151617],
#     'title': ['...', '...', '...', '...'],
#     'abstract': ['...', '...', '...', '...']
# })



GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
CSV_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv"
filename = CSV_PATH
df = pd.read_csv(filename)


def save_gold_dict(gold_dict, filename="gold_dict.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(gold_dict, f)

def load_gold_dict(filename=GOLD_DICT_PATH):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):  # No previous gold_dict or empty file
        return {}


# Utility to fetch metadata from folders
def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(path_to_dirs, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}.txt")
    with open(metadata_file_path, 'r') as f:
        metadata = f.read()
    return metadata

# Main game
def play_game(df):
    # Load gold_dict from a previous game or start with an empty one
    gold_dict = load_gold_dict()
    wrong_dict = {}
    biomes_df = df.groupby('biome')

    # Count gold_dict samples for each biome
    gold_biome_counts = {biome: sum(1 for _, biome_val in gold_dict.values() if biome_val == biome) for biome in biomes_df.groups}

    # Continue the game until all samples are processed
    while True:
        # Select the biome with the least number of samples in gold_dict
        prioritized_biome = min(gold_biome_counts, key=gold_biome_counts.get)
        group = biomes_df.get_group(prioritized_biome)
        
        # If we've already processed all samples in this biome, update its count to a large value and move to the next biome
        if gold_biome_counts[prioritized_biome] >= len(group):
            gold_biome_counts[prioritized_biome] = float('inf')
            continue

        row = group.iloc[gold_biome_counts[prioritized_biome]]

        # Retrieve and display metadata
        metadata = fetch_metadata_from_sample(row['sample'])
        print(metadata)

        # Display title and abstract
        print(f"Title: {row['title']}")
        print(f"Abstract: {row['abstract']}")

        # Ask first question
        ans = input("Is the pmid informative of the sample nature? (y/n): ")
        if ans == 'n':
            wrong_dict[row['sample']] = row['pmid']
            continue

        # Ask second question
        ans = input(f"Is the biome of this sample: {prioritized_biome}? (y/n): ")
        if ans == 'y':
            gold_dict[row['sample']] = (row['pmid'], prioritized_biome)
            save_gold_dict(gold_dict)
            gold_biome_counts[prioritized_biome] += 1
            continue

        # Ask for correct biome if the answer was 'n'
        biome_input = input("Which biome is it? (a for animal, w for water, s for soil, p for plant, l for lab, u for unknown): ")
        biome_mapping = {
            'a': 'animal',
            'w': 'water',
            's': 'soil',
            'p': 'plant',
            'l': 'lab',
            'u': 'unknown'
        }
        gold_dict[row['sample']] = (row['pmid'], biome_mapping.get(biome_input, 'unknown'))
        save_gold_dict(gold_dict)

        # Increment the count for the prioritized_biome
        gold_biome_counts[prioritized_biome] += 1

    return gold_dict, wrong_dict


gold_dict, wrong_dict = play_game(df)
print("Gold Dictionary:", gold_dict)
print("Wrong Dictionary:", wrong_dict)













# =============================================================================
# options = ['DRS001007', 'DRS001008', 'DRS001009', 'DRS001010'] 
# 
# # selecting rows based on condition 
# rslt_df = df.loc[df['sample'].isin(options)] 
# =============================================================================




