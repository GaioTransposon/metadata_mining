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
GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
CSV_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/training_data_pmids_based.csv"
filename = CSV_PATH
df = pd.read_csv(filename)


def save_gold_data(data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_gold_data(filename=GOLD_DICT_PATH):
    try:
        with open(filename, "rb") as f:
            gold_dict = pickle.load(f)
        processed_pmids = {value[0] for value in gold_dict.values()}
        return gold_dict, processed_pmids
    except (FileNotFoundError, EOFError):  
        return {}, set()



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
        count = sum(1 for values in gold_dict.values() if values[1] == biome)
        print(f"{biome.capitalize()}: {count}")
        
        
def play_game(df):
    df['biome_gpt'] = df['biome_gpt'].replace('human', 'animal')
    df['biome'] = df['biome'].replace('aquatic', 'water')
    
    gold_dict, processed_pmids = load_gold_data()
    
    print("Unique biome values in DataFrame:", df['biome'].unique())

    biomes_df = df.groupby('biome')

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

    gold_biome_count = sum(1 for value in gold_dict.values() if value[1] == selected_biome)

    while gold_biome_count < len(group):
        row = group.iloc[gold_biome_count]
        if row['pmid'] in processed_pmids:
            gold_biome_count += 1
            continue
        
        metadata = fetch_metadata_from_sample(row['sample'])
        print('\n\n\n\n\n\n\n\n')
        print(metadata)

        ans = input(f"Is the biome of this sample: {selected_biome}? (y/n/q): ")
        
        if ans.lower() == 'q':
            display_biome_counts(gold_dict)
            print("Exiting game...")
            save_gold_data(gold_dict)
            break

        processed_pmids.add(row['pmid'])
        
        if ans == 'y':
            gold_dict[row['sample']] = (row['pmid'], selected_biome)
            save_gold_data(gold_dict)
            gold_biome_count += 1
            continue

        biome_input = input("Which biome is it? (a for animal, w for water, s for soil, p for plant, o for other): ")
        gold_dict[row['sample']] = (row['pmid'], biome_mapping.get(biome_input, 'other'))
        save_gold_data(gold_dict)

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








