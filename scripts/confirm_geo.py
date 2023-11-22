#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:18:27 2023

@author: dgaio
"""

# =============================================================================
# import os
# import re
# 
# 
# patterns = [
#     r'[^a-zA-Z]lat[^a-zA-Z]\w*', # Matches surrounded by non-letter characters
#     r'[^a-zA-Z]lon[^a-zA-Z]\w*',  
#     r'[^a-zA-Z]latitude[^a-zA-Z]\w*',  
#     r'[^a-zA-Z]longitude[^a-zA-Z]\w*',  
#     r'[^a-zA-Z]coordinates[^a-zA-Z]\w*', 
#     r'[^a-zA-Z]coordinate[^a-zA-Z]\w*',  
#     r'[^a-zA-Z]location[^a-zA-Z]\w*',
#     r'[^a-zA-Z]geograph\w*',  # Matches with non-letter characters left of the word
#     r'[^a-zA-Z]geo\w*',
# ]
# 
# 
# 
# def extract_coordinates(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
# 
#     sample_name = lines[0].strip()
#     coordinate_lines = []
# 
#     for line in lines:
#         if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
#             coordinate_lines.append(f"{sample_name}\n{line.strip()}\n")
# 
#     return coordinate_lines
# 
# 
# 
# 
# path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
# 
# # Assuming the directories are named as 'dir_XXX'
# dir_names = [name for name in os.listdir(path_to_dirs) if name.startswith('dir_80')]
# 
# output_file = path_to_dirs+"/samples_coordinates.txt"
# 
# with open(output_file, 'w') as outfile:
#     for dir_name in dir_names:
#         dir_path = os.path.join(path_to_dirs, dir_name)
#         for filename in os.listdir(dir_path):
#             if filename.endswith('.txt'):  # or other specific extension if needed
#                 file_path = os.path.join(dir_path, filename)
#                 coordinate_lines = extract_coordinates(file_path)
#                 outfile.writelines(coordinate_lines)
# 
# 
# 
# # remove lines that do not contain any digit
# # remove lines that are over n tokens? like abstracts? or extract to part of surrounding text
# =============================================================================

   

import os
import pickle
import random
import re




path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"



patterns = [
    r'[^a-zA-Z]lat[^a-zA-Z]\w*', # Matches surrounded by non-letter characters
    r'[^a-zA-Z]lon[^a-zA-Z]\w*', 
    r'[^a-zA-Z]geo[^a-zA-Z]\w*', 
    r'[^a-zA-Z]latitude[^a-zA-Z]\w*',  
    r'[^a-zA-Z]longitude[^a-zA-Z]\w*',  
    r'[^a-zA-Z]coordinates[^a-zA-Z]\w*', 
    r'[^a-zA-Z]coordinate[^a-zA-Z]\w*',  
    r'[^a-zA-Z]location[^a-zA-Z]\w*',
    r'[^a-zA-Z]geograph\w*',  # Matches with non-letter characters left of the word
    r'[^a-zA-Z]geo\w*',
]



def save_gold_data(gold_data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(gold_data, f)

def update_gold_data(sample_id, coordinates, location_text, gold_data):
    gold_dict, processed_pmids = gold_data
    gold_dict[sample_id] = (
        gold_dict[sample_id][0],  # Keep the original pmid
        gold_dict[sample_id][1],  # Keep the original biome
        coordinates,
        location_text
    )
    save_gold_data(gold_data)

    
# Utility to fetch metadata from folders
def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(path_to_dirs, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")

    relevant_lines = []
    with open(metadata_file_path, 'r') as f:
        for line in f:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                relevant_lines.append(line.strip())

    return '\n'.join(relevant_lines)


def display_biome_stats(gold_dict):
    biome_counts = {biome: 0 for biome in ['animal', 'plant', 'water', 'soil', 'unknown']}
    for details in gold_dict.values():
        if len(details) > 3:
            biome_counts[details[1]] += 1

    print("\nNumber of samples classified (more than two values) per biome:")
    print("------------------------------------------------------------")
    for biome, count in biome_counts.items():
        print(f"{biome.capitalize()}: {count}")




def play_game(gold_data):
    gold_dict, _ = gold_data
    biome_mapping = {
        'a': 'animal',
        'w': 'water',
        'p': 'plant',
        's': 'soil'
    }

    biome_input = input("\nWhich biome do you want to focus on? (a for animal, w for water, p for plant, s for soil, q to quit): ").lower()
    if biome_input == 'q':
        display_biome_stats(gold_dict)
        print("Exiting game...")
        return

    selected_biome = biome_mapping.get(biome_input)
    if not selected_biome:
        print("Invalid option. Please choose a valid biome.")
        return

    biome_samples = [sample for sample, data in gold_dict.items() if data[1] == selected_biome]
    if not biome_samples:
        print("No samples found for this biome.")
        return

    while True:
        sample = random.choice(biome_samples)
        
        # Print the sample ID before fetching the metadata
        print(f"\n>{sample}")
        metadata = fetch_metadata_from_sample(sample)
        print(metadata)

        coordinates_input = input("What are the coordinates of the sample? (Enter 'NA' if not available or 'q' to quit): ")
        if coordinates_input.lower() == 'q':
            break
        coordinates = 'geo: ' + coordinates_input

        location_text_input = input("Describe the location of the sample in text (Enter 'NA' if not available or 'q' to quit): ")
        if location_text_input.lower() == 'q':
            break
        location_text = 'geo_text: ' + location_text_input

        update_gold_data(sample, coordinates, location_text, gold_data)
        print("Information saved successfully.")

    display_biome_stats(gold_dict)
    print("Exiting game...")





# Code to initiate the game
try:
    with open(GOLD_DICT_PATH, "rb") as f:
        gold_data = pickle.load(f)
except (FileNotFoundError, EOFError):
    gold_data = ({}, set())

play_game(gold_data)






