#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:45:43 2024

@author: dgaio
"""



import os
import pickle

GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
METADATA_DIRECTORY = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"

# Function to fetch metadata
def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(METADATA_DIRECTORY, folder_name)
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as file:
        return file.read()

# Load the existing data
with open(GOLD_DICT_PATH, 'rb') as file:
    data, processed_pmids = pickle.load(file)

# Ask the user for the sample key
sample_key = input("Enter the sample key to retrieve data for: ")

# Check if the key exists in the dictionary
if sample_key in data:
    # Retrieve and print the metadata
    metadata = fetch_metadata_from_sample(sample_key)
    print(f"Metadata for '{sample_key}':\n{metadata}")

    # Retrieve the current tuple for the key and display the values
    values = data[sample_key]  # Accessing tuple directly since no modification is needed
    print(f"Current values for '{sample_key}': {values}")
else:
    print(f"No data found for the sample key '{sample_key}'.")
