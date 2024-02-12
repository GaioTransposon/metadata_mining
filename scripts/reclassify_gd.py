#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:36:40 2024

@author: dgaio
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:15:33 2024

@author: dgaio
"""

import os
import pickle

# Paths
GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
path_to_dirs = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"

# Parameters for reclassification
current_biome_to_search_in = 'water'  # The biome to search within for the keyword
word_to_search = 'bioreactor'       # The keyword to search for in the metadata
new_biome_to_assign = 'other'        # The new biome to assign if the keyword is found

def load_gold_data(filename=GOLD_DICT_PATH):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}, set()

def save_gold_data(data, filename=GOLD_DICT_PATH):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(path_to_dirs, folder_name)
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    try:
        with open(metadata_file_path, 'r') as f:
            metadata = f.read()
        return metadata
    except FileNotFoundError:
        return "Metadata not found."

def reclassify_samples():
    gold_dict, processed_pmids = load_gold_data()
    print(gold_dict)

    for sample, values in gold_dict.items():
        pmid, biome = values[0], values[1]  # Extracting the first two elements: PMID and biome
        if biome == current_biome_to_search_in:
            metadata = fetch_metadata_from_sample(sample)
            if word_to_search in metadata.lower():
                print(f"\nSample {sample} with biome '{biome}' contains '{word_to_search}' in its metadata.")
                print("\nMetadata text:")
                print(metadata)
                ans = input(f"\nShould this sample be reclassified as '{new_biome_to_assign}'? (y/n/q to quit): ")
                if ans.lower() == 'q':
                    print("Quitting and saving changes...")
                    save_gold_data((gold_dict, processed_pmids))
                    return  # Exit the function gracefully
                elif ans.lower() == 'y':
                    # Create a new tuple with the new biome, keeping the rest of the information unchanged
                    new_values = (pmid, new_biome_to_assign) + values[2:]
                    gold_dict[sample] = new_values
                    print(f"Sample {sample} reclassified as '{new_biome_to_assign}'.")
                else:
                    print(f"No changes made for sample {sample}.")

    save_gold_data((gold_dict, processed_pmids))
    print("\nGold dictionary updated with any reclassifications.")

reclassify_samples()


