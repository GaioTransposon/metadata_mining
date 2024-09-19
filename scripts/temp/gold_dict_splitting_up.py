#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:04:31 2024

@author: dgaio
"""

import pickle

# Load the existing dictionary from a pickle file
GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
with open(GOLD_DICT_PATH, 'rb') as file:
    gold_dict = pickle.load(file)
    gold_dict = gold_dict[0]

# Create a new dictionary to store entries with more than 2 values
gold_dict_biomesonly = {}

# Iterate through the original dictionary
for key, value in list(gold_dict.items()):  # Using list to avoid modifying the dict while iterating
    if len(value) <= 2:
        gold_dict_biomesonly[key] = value  # Move entry to the new dictionary
        del gold_dict[key]  # Remove entry from the original dictionary

len(gold_dict_biomesonly)
len(gold_dict)

# Save the modified original dictionary back to the same file
with open(GOLD_DICT_PATH, 'wb') as file:
    pickle.dump(gold_dict, file)

# Save the filtered dictionary to a new file
with open("/Users/dgaio/github/metadata_mining/source_data/gold_dict_biomesonly.pkl", 'wb') as file:
    pickle.dump(gold_dict_biomesonly, file)



# mind: the processed pmids have not being carried along (neither in gold dict 
# nor in the new one, because they are matching 100% with the first value of 
# each key, so the processed pmid list was not carrying any extra value)





