#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:43:50 2023

@author: dgaio
"""


import os
import pickle

GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
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
sample_key = input("What is the sample key you want to edit? ")

# Check if the key exists in the dictionary
if sample_key in data:
    # Retrieve and print the metadata
    metadata = fetch_metadata_from_sample(sample_key)
    print(f"Metadata for '{sample_key}':\n{metadata}")

    # Retrieve the current tuple for the key
    values = list(data[sample_key])  # Convert tuple to list to allow modifications
    print(f"Current values for '{sample_key}': {values}")

    # Ask the user which value to change
    value_position = input("Which value position do you want to change? Enter a number: ")
    if value_position.isdigit() and 0 <= int(value_position) < len(values):
        value_position = int(value_position)
        print(f"Current value at position {value_position}: {values[value_position]}")

        # Ask for the new value
        new_value = input("Enter the new value: ")
        values[value_position] = new_value
        data[sample_key] = tuple(values)  # Convert back to tuple and reassign to the key

        # Save the updated dictionary back to the .pkl file
        with open(GOLD_DICT_PATH, 'wb') as file:
            pickle.dump((data, processed_pmids), file)

        print(f"Value at position {value_position} for '{sample_key}' has been updated to '{new_value}'.")
    else:
        print("Invalid position number.")

else:
    print(f"The key '{sample_key}' was not found in the dictionary.")




# =============================================================================
# replacing all unknown occurrences to other
# 
# import os
# import pickle
# 
# GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"
# METADATA_DIRECTORY = "/path/to/metadata/directory"  # Replace with the actual directory path
# 
# # Function to fetch metadata
# def fetch_metadata_from_sample(sample):
#     folder_name = f"dir_{sample[-3:]}"
#     folder_path = os.path.join(METADATA_DIRECTORY, folder_name)
#     metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
#     with open(metadata_file_path, 'r') as file:
#         return file.read()
# 
# # Load the existing data
# with open(GOLD_DICT_PATH, 'rb') as file:
#     data, processed_pmids = pickle.load(file)
#     
# 
# # Function to replace 'unknown' with 'other' in a value
# def replace_unknown_with_other(value):
#     return tuple('other' if isinstance(v, str) and v.lower() == 'unknown' else v for v in value)
# 
# # Replace 'unknown' with 'other' in the entire dictionary
# data = {key: replace_unknown_with_other(value) for key, value in data.items()}
# 
# 
# 
# 
# # Function to print the head and tail of the dictionary
# def print_head_and_tail(dictionary, n=5):
#     keys = list(dictionary.keys())
#     head_keys = keys[:n]
#     tail_keys = keys[-n:]
# 
#     print("Head of the dictionary:")
#     for key in head_keys:
#         print(f"{key}: {dictionary[key]}")
# 
#     print("\nTail of the dictionary:")
#     for key in tail_keys:
#         print(f"{key}: {dictionary[key]}")
# 
# # Print the head and tail of the dictionary
# print_head_and_tail(data)
# 
# 
# # Save the updated dictionary back to the .pkl file
# with open(GOLD_DICT_PATH, 'wb') as file:
#     pickle.dump((data, processed_pmids), file)
# 
# print("All occurrences of 'unknown' have been replaced with 'other' in the dictionary.")
# 
# 
# =============================================================================

################################################################################
################################################################################

# =============================================================================
# # MORE DRASTIC! 
# # to remove all 3rd and 4th values 
# 
# 
# # Function to load gold data
# def load_gold_data(filename):
#     with open(filename, "rb") as f:
#         return pickle.load(f)
# 
# # Function to save gold data
# def save_gold_data(data, filename):
#     with open(filename, "wb") as f:
#         pickle.dump(data, f)
# 
# # Load the gold data
# gold_data = load_gold_data(GOLD_DICT_PATH)
# 
# # Modify each entry in the gold_dict
# gold_dict, processed_pmids = gold_data
# for sample_id, details in gold_dict.items():
#     # Check if the entry has more than 3 values and then modify it
#     if len(details) > 3:
#         print(details)
#         gold_dict[sample_id] = details[:3]  # Keep only the first three values
# 
# # Save the modified gold data back to the pickle file
# save_gold_data(gold_data, GOLD_DICT_PATH)
# 
# print("Updated gold_dict and saved to the pickle file.")
# =============================================================================
