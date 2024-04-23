#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:02:40 2024

@author: dgaio
"""

import os
import pickle
import random

path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"

def save_gold_data(gold_data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(gold_data, f)

def update_gold_data(sample_id, sub_biome, gold_data):
    gold_dict, processed_pmids = gold_data
    current_data = list(gold_dict[sample_id])
    current_data[2] = sub_biome  # Update only the sub-biome (third value)
    gold_dict[sample_id] = tuple(current_data)
    save_gold_data(gold_data)

# Utility to fetch metadata from folders
def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(path_to_dirs, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as f:
        metadata = f.read()
    return metadata

def display_biome_stats(gold_dict):
    # Initialize biome_counts dynamically based on the existing biomes in gold_dict
    biome_counts = {}
    for details in gold_dict.values():
        if len(details) > 3 and 'placeholder' not in details:  # Skip placeholder sub-biomes
            biome = details[1]  # Biome type is expected to be at index 1
            if biome not in biome_counts:
                biome_counts[biome] = 0  # Initialize count if biome key doesn't exist
            biome_counts[biome] += 1  # Safely increment biome count

    print("\nNumber of samples classified (more than three values) per biome, excluding 'placeholder' sub-biomes:")
    print("------------------------------------------------------------")
    for biome, count in biome_counts.items():
        print(f"{biome.capitalize()}: {count}")



def play_game(gold_data):
    gold_dict, _ = gold_data
    biome_mapping = {
        'a': 'animal',
        'w': 'water',
        'p': 'plant',
        's': 'soil',
        'o': 'other'
    }

    while True:
        biome_input = input("\nWhich biome do you want to focus on? (a for animal, w for water, p for plant, s for soil, o for other, q to quit): ").lower()

        if biome_input == 'q':
            display_biome_stats(gold_dict)
            print("Exiting game...")
            break

        selected_biome = biome_mapping.get(biome_input)
        if not selected_biome:
            print("Invalid option. Please choose a valid biome.")
            continue

        biome_samples = [sample for sample, data in gold_dict.items() if len(data) > 2 and data[1] == selected_biome and data[2] == 'placeholder']

        if not biome_samples:
            print("No samples with 'placeholder' sub-biome found for this biome.")
            continue

        sample = random.choice(biome_samples)
        print(f"\n>{sample}")
        metadata = fetch_metadata_from_sample(sample)
        print(metadata)

        sub_biome = input("\nWhich sub-biome does this sample come from? (Enter 'q' to quit): ")
        if sub_biome == 'q':
            break

        update_gold_data(sample, sub_biome, gold_data)
        print("Sub-biome information updated successfully.")




# =============================================================================
# def play_game(gold_data):
#     gold_dict, _ = gold_data
#     biome_mapping = {
#         'a': 'animal',
#         'w': 'water',
#         'p': 'plant',
#         's': 'soil',
#         'o': 'other'
#     }
# 
#     selected_biome = None  # Initialize selected biome variable
# 
#     while True:
#         if not selected_biome:
#             biome_input = input("\nWhich biome do you want to focus on? (a for animal, w for water, p for plant, s for soil, o for other, q to quit): ").lower()
# 
#             if biome_input == 'q':
#                 display_biome_stats(gold_dict)
#                 print("Exiting game...")
#                 break
# 
#             selected_biome = biome_mapping.get(biome_input)
#             if not selected_biome:
#                 print("Invalid option. Please choose a valid biome.")
#                 continue
# 
#         # Select only samples where the third value is 'placeholder' and the biome matches the selected biome
#         biome_samples = [sample for sample, data in gold_dict.items() if len(data) > 2 and data[1] == selected_biome and data[2] == 'placeholder']
# 
#         if not biome_samples:
#             print("No samples with 'placeholder' sub-biome found for this biome.")
#             selected_biome = None  # Reset selected biome if no samples are found
#             continue
# 
#         sample = random.choice(biome_samples)
#         print(f"\n>{sample}")
#         metadata = fetch_metadata_from_sample(sample)
#         print(metadata)
# 
#         sub_biome = input("\nWhich sub-biome does this sample come from? (Enter 'q' to quit): ")
#         if sub_biome == 'q':
#             break
# 
#         update_gold_data(sample, sub_biome, gold_data)
#         print("Sub-biome information updated successfully.")
# =============================================================================


# Code to initiate the game
try:
    with open(GOLD_DICT_PATH, "rb") as f:
        gold_data = pickle.load(f)
except (FileNotFoundError, EOFError):
    gold_data = ({}, set())

play_game(gold_data)












