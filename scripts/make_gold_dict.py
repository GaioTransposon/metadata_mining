#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:46:32 2024

@author: dgaio
"""


# run as:
# python ~/github/metadata_mining/scripts/make_gold_dict.py

import pandas as pd
import os
import pickle
import random


# ------------------------------------------------------------------
# All data live in the current working directory (cwd).
#   • In Docker:  /MicrobeAtlasProject  (set by WORKDIR)
#   • On host:    ~/MicrobeAtlasProject (if you run locally)
# ------------------------------------------------------------------
WORK_DIR      = os.getcwd()                    # usually /MicrobeAtlasProject
CSV_PATH      = os.path.join(WORK_DIR, "training_data_pmids_based.csv")
GOLD_DICT_PATH = os.path.join(WORK_DIR, "gold_dict.pkl")
SPLIT_DIR      = os.path.join(WORK_DIR, "sample_info_split_dirs")


# Now you can continue with the rest of the script
df = pd.read_csv(CSV_PATH)


def save_gold_data(data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_gold_data(filename=GOLD_DICT_PATH):
    try:
        with open(filename, "rb") as f:
            gold_dict = pickle.load(f)
        return gold_dict
    except (FileNotFoundError, EOFError):  
        return {}
    

def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(SPLIT_DIR, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as f:
        metadata = f.read()
    return metadata


def display_biome_stats(gold_dict):
    biome_counts = {biome: 0 for biome in ['animal', 'plant', 'water', 'soil', 'other']}
    for details in gold_dict.values():
        if len(details) == 5:
            biome_counts[details[1]] += 1

    print("\nNumber of samples classified (5 values per sample):")
    print("------------------------------------------------------------")
    for biome, count in biome_counts.items():
        print(f"{biome.capitalize()}: {count}")



def play_game(df):
    
    df['biome'] = df['biome'].replace('aquatic', 'water')
    gold_dict = load_gold_data()
    biome_mapping = {
        'a': 'animal',
        'w': 'water',
        's': 'soil',
        'p': 'plant',
        'o': 'other'
    }

    while True:
        biome_input = input("\nChoose a biome to focus on (a: animal, w: water, s: soil, p: plant, o: other, q: quit): ").lower()
        if biome_input == 'q':
            display_biome_stats(gold_dict)
            save_gold_data(gold_dict)
            print("Exiting game...")
            break

        selected_biome = biome_mapping.get(biome_input)
        if not selected_biome:
            print("Invalid option. Please choose a valid biome.")
            continue

        # pick random sample from selected biome 
        sample_id = df[df['biome'] == selected_biome].sample(1)['sample'].values[0]
        print(f"Selected Sample: {sample_id}")
        metadata = fetch_metadata_from_sample(sample_id)
        print('\n\n\n\n\n\n\n\n')
        print(metadata)

        biome_confirmation = input(f"Confirm the main biome as {selected_biome} (y/n/q): ").lower()
        if biome_confirmation == 'q':
            break
        elif biome_confirmation == 'n':
            new_biome_input = input("Enter the correct biome (a: animal, w: water, s: soil, p: plant, o: other, q: quit): ").lower()
            if new_biome_input == 'q':
                break
            new_biome = biome_mapping.get(new_biome_input)
            if not new_biome:
                print("Invalid biome. Try again.")
                continue
            selected_biome = new_biome

        sub_biome = input("Enter the sub-biome (or type 'q' to exit): ")
        if sub_biome == 'q':
            break

        coordinates = input("Enter the coordinates of the sample (or type 'q' to exit): ")
        if coordinates == 'q':
            break

        location_text = input("Describe the location of the sample in text (or type 'q' to exit): ")
        if location_text == 'q':
            break

        # save only if all fields are filled
        if all([sub_biome, coordinates, location_text]):
            gold_dict[sample_id] = (sample_id, selected_biome, sub_biome, coordinates, location_text)
            save_gold_data(gold_dict)
            print("Information saved successfully.")
        else:
            print("Incomplete data. Nothing was saved.")


play_game(df)




