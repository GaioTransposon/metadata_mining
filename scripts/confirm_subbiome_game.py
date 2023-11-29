#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:48:55 2023

@author: dgaio
"""


# # run as: 
# python ~/github/metadata_mining/scripts/game2.py 
   

import os
import pickle
import random

path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"

def save_gold_data(gold_data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(gold_data, f)

def update_gold_data(sample_id, sub_biome, coordinates, location_text, gold_data):
    gold_dict, processed_pmids = gold_data
    gold_dict[sample_id] = (
        gold_dict[sample_id][0],  # Keep the original pmid
        gold_dict[sample_id][1],  # Keep the original biome
        sub_biome,
        coordinates,
        location_text
    )
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
    biome_counts = {biome: 0 for biome in ['animal', 'plant', 'water', 'soil', 'unknown']}
    for details in gold_dict.values():
        if len(details) > 2:
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

    while True:
        biome_input = input("\nWhich biome do you want to focus on? (a for animal, w for water, p for plant, s for soil, q to quit): ").lower()

        if biome_input == 'q':da
            display_biome_stats(gold_dict)
            print("Exiting game...")
            break

        selected_biome = biome_mapping.get(biome_input)
        if not selected_biome:
            print("Invalid option. Please choose a valid biome.")
            continue

        biome_samples = [sample for sample, data in gold_dict.items() if data[1] == selected_biome]
        
        if not biome_samples:
            print("No samples found for this biome.")
            continue

        sample = random.choice(biome_samples)
        
        # Print the sample ID before fetching the metadata
        print(f"\n>{sample}")
        metadata = fetch_metadata_from_sample(sample)

        print(metadata)

        sub_biome = input("\nWhich sub-biome does this sample come from? (Enter 'NA' if not available or 'q' to quit): ")
        if sub_biome == 'q':
            break

        coordinates = input("What are the coordinates of the sample? (Enter 'NA' if not available or 'q' to quit): ")
        if coordinates == 'q':
            break

        location_text = input("Describe the location of the sample in text (Enter 'NA' if not available or 'q' to quit): ")
        if location_text == 'q':
            break

        update_gold_data(sample, sub_biome, coordinates, location_text, gold_data)

        print("Information saved successfully.")


# Code to initiate the game
try:
    with open(GOLD_DICT_PATH, "rb") as f:
        gold_data = pickle.load(f)
except (FileNotFoundError, EOFError):
    gold_data = ({}, set())

play_game(gold_data)


