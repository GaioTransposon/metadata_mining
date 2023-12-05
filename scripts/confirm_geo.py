#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:18:27 2023

@author: dgaio
"""


import os
import pickle
import pandas as pd
import random

path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
JOAO_BIOMES_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/joao_biomes_parsed.csv"

# Load the CSV file
joao_biomes_df = pd.read_csv(JOAO_BIOMES_PATH)
# Filter out rows with NaN coordinates
joao_biomes_df = joao_biomes_df.dropna(subset=['coordinates'])

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


def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(path_to_dirs, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")

    with open(metadata_file_path, 'r') as f:
        return f.read()


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
        's': 'soil',
        'u': 'unknown'
    }

    # # Filter gold_dict samples based on joao_biomes_df
    # filtered_samples = set(joao_biomes_df['sample'])
    # gold_dict = {k: v for k, v in gold_dict.items() if k in filtered_samples}

    biome_input = input("\nWhich biome do you want to focus on? (a for animal, w for water, p for plant, s for soil, u for unknown, q to quit): ").lower()
    
    if biome_input == 'q':
        display_biome_stats(gold_dict)
        print("Exiting game...")
        return

    selected_biome = biome_mapping.get(biome_input)
    if not selected_biome:
        print("Invalid option. Please choose a valid biome.")
        return

    # Filter out samples that already have four values
    biome_samples = {sample for sample, data in gold_dict.items() if data[1] == selected_biome and len(data) < 4}

    # Set to track samples handled in the current session
    handled_samples = set()

    while biome_samples - handled_samples:
        sample = random.choice(list(biome_samples - handled_samples))
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

        # Add the sample to the set of handled samples
        handled_samples.add(sample)

    # Save and reload the gold_dict before displaying stats
    save_gold_data(gold_data)
    try:
        with open(GOLD_DICT_PATH, "rb") as f:
            gold_data = pickle.load(f)
            display_biome_stats(gold_data[0])
    except (FileNotFoundError, EOFError):
        print("Error in loading the updated gold data.")

    print("Exiting game...")



# Code to initiate the game
try:
    with open(GOLD_DICT_PATH, "rb") as f:
        gold_data = pickle.load(f)
except (FileNotFoundError, EOFError):
    gold_data = ({}, set())

play_game(gold_data)











