#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:55:43 2023

@author: dgaio
"""


# # run as: 
# python ~/github/metadata_mining/scripts/confirm_biome_game.py 
    

import pandas as pd
import os
import pickle


path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
CSV_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/training_data_pmids_based.csv"
filename = CSV_PATH
df = pd.read_csv(filename)


def save_gold_data(data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_gold_data(filename=GOLD_DICT_PATH):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):  # No previous data or empty file
        return {}, set()

# Utility to fetch metadata from folders
def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(path_to_dirs, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as f:
        metadata = f.read()
    return metadata

def display_biome_counts(gold_dict):
    biomes = ['animal', 'plant', 'water', 'soil', 'other']
    print("\nBiome Counts in Gold Dictionary:")
    print("---------------------------------")
    for biome in biomes:
        count = sum(1 for values in gold_dict.values() if len(values) >= 2 and values[1] == biome)
        print(f"{biome.capitalize()}: {count}")


def play_game(df):
    
    df['biome_gpt'] = df['biome_gpt'].replace('human', 'animal')
    df['biome'] = df['biome'].replace('aquatic', 'water')
    
    # Load gold_dict and processed_pmids from a previous game or start with an empty ones
    gold_dict, processed_pmids = load_gold_data()
    
    # Debug: Print unique values in 'biome' column
    print("Unique biome values in DataFrame:", df['biome'].unique())

    # Group by biome_gpt
    biomes_df = df.groupby('biome')

    # Ask user which biome to focus on
    biome_input = input("Which biome do you want to focus on? (a for animal, w for water, s for soil, p for plant, o for other): ")
    biome_mapping = {
        'a': 'animal',
        'w': 'water',
        's': 'soil',
        'p': 'plant',
        'o': 'other'
    }
    
    selected_biome = biome_mapping.get(biome_input, 'other')
    group = biomes_df.get_group(selected_biome)

    # Count gold_dict samples for the selected biome
    gold_biome_count = sum(1 for value in gold_dict.values() if value[1] == selected_biome)
    #gold_biome_count = sum(1 for _, biome_val in gold_dict.values() if biome_val == selected_biome)

    # Continue the game until all samples in the selected biome are processed
    while gold_biome_count < len(group):
        row = group.iloc[gold_biome_count]
        if row['pmid'] in processed_pmids:
            gold_biome_count += 1
            continue
        
        # Retrieve and display metadata
        metadata = fetch_metadata_from_sample(row['sample'])
        print('\n\n\n\n\n\n\n\n')
        print(metadata)

        # Ask the user
        ans = input(f"Is the biome of this sample: {selected_biome}? (y/n/q): ")
        
        # Allow the user to quit
        if ans.lower() == 'q':
            display_biome_counts(gold_dict)
            print("Exiting game...")
            save_gold_data((gold_dict, processed_pmids))
            break

        processed_pmids.add(row['pmid'])
        
        if ans == 'y':
            gold_dict[row['sample']] = (row['pmid'], selected_biome)
            save_gold_data((gold_dict, processed_pmids))
            gold_biome_count += 1
            continue

        # Ask for correct biome if the answer was 'n'
        biome_input = input("Which biome is it? (a for animal, w for water, s for soil, p for plant, o for other): ")
        gold_dict[row['sample']] = (row['pmid'], biome_mapping.get(biome_input, 'other'))
        save_gold_data((gold_dict, processed_pmids))

        # Increment the count for the selected biome
        gold_biome_count += 1

    return gold_dict



gold_dict = play_game(df)
print("Gold Dictionary:", gold_dict)



# =============================================================================
# options = [35258340] 
# 
# # selecting rows based on condition 
# rslt_df = df.loc[df['pmid'].isin(options)] 
# =============================================================================








