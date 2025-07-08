#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:54:35 2025

@author: danielagaio
"""

import pandas as pd
import pickle
import os


def load_gold_dict(file_path):
    """Load the gold standard dictionary from a pickle file."""
    with open(file_path, 'rb') as file:
        gold_dict = pickle.load(file)
    return gold_dict

def read_data(file_path):
    """Read and preprocess the data file."""
    df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)
    # Check for duplicates
    print(f"Number of total entries: {len(df)}")
    print(f"Number of unique sample IDs: {df['col_0'].nunique()}")
    return df

def check_biomes(df, gold_dict):
    """Compare biomes from the data against the gold standard and return mismatches."""
    mismatches = []
    for idx, row in df.iterrows():
        sample_id = row['col_0'].strip()  # Normalize data by stripping whitespace
        biome = row['col_1'].strip().lower()  # Normalize and make case insensitive

        # Access the biome from the tuple and normalize it
        gold_biome = gold_dict.get(sample_id, ('Not found',))[1].strip().lower() if sample_id in gold_dict else 'Not found'

        # Debug output to check what is being compared
        print(f"Checking sample ID: {sample_id}, Biome: {biome}, Gold Dict Biome: {gold_biome}")

        if gold_biome != biome:  # Case insensitive comparison
            mismatches.append(sample_id)
    return mismatches




# Load gold standard dictionary
home_dir = os.getenv('HOME')
gold_dict_path = os.path.join(home_dir, "MicrobeAtlasProject/gold_dict.pkl") 
gold_dict = load_gold_dict(gold_dict_path)
data_file_path = os.path.join(home_dir, "MicrobeAtlasProject/GH_collect_output_here.txt") 




# Load gold standard dictionary
gold_dict = load_gold_dict(gold_dict_path)

# Load and check the data file
data_df = read_data(data_file_path)

# Get samples with incorrect biomes
incorrect_samples = check_biomes(data_df, gold_dict)

# Output incorrect samples
print("Samples with incorrect biomes:", incorrect_samples)
print(f"Number of incorrect samples: {len(incorrect_samples)}")





def extract_incorrect_samples(file_path, incorrect_samples):
    """
    Extract entries for incorrectly classified samples from a text file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the file into sections based on a unique pattern that denotes a new sample
    sections = content.split("-----\n")
    
    # Filter sections to keep only those with sample IDs that are in the incorrect_samples list
    filtered_sections = [
        section for section in sections
        if any(f'sample_ID={sample_id}' in section for sample_id in incorrect_samples)
    ]
    
    # Join the filtered sections back into a single string with separator
    return "-----\n".join(filtered_sections)

# Path to the text file
file_path = os.path.join(home_dir, "MicrobeAtlasProject/metadata_chunks_202407191703.txt")  

# Extract sections for incorrect samples
incorrect_sample_sections = extract_incorrect_samples(file_path, incorrect_samples)

# Optionally, write the filtered sections to a new file


with open(os.path.join(home_dir, "MicrobeAtlasProject/GH_incorrect_samples.txt"), 'w') as output_file:
    output_file.write(incorrect_sample_sections)

print("Filtered sections saved to 'GH_incorrect_samples.txt'")





