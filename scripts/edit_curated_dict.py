#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:43:50 2023

@author: dgaio
"""

import pickle


GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"


with open(GOLD_DICT_PATH, 'rb') as file:
    data, processed_pmids = pickle.load(file)



# =============================================================================
# for key, details in data.items():
#     if len(details) > 2:
#         print(f"Key: '{key}', Values: {details}")
# =============================================================================




# Define the key for which you want to change the third value
key_to_edit = 'SRS4847591'
new_value = 'plastic'
value_n = 2 # 0 : PMID; 1 : biome ; 2 : sub-biome ; 3 : coordinates ; 4 : geo-loc

# Check if the key exists in the dictionary
if key_to_edit in data:
    # Retrieve the current tuple for the key
    values = list(data[key_to_edit])  # Convert tuple to list to allow modifications
    
    print(f"The value for key '{key_to_edit}' was {values[value_n]}... ")

    values[value_n] = new_value  
    data[key_to_edit] = tuple(values)  # Convert back to tuple and reassign to the key
    print('\n')
    print(data[key_to_edit])
    
    # Save the updated dictionary back to the .pkl file
    with open(GOLD_DICT_PATH, 'wb') as file:
        pickle.dump((data, processed_pmids), file)
    
    print(f" \n ... and has been updated to '{new_value}'.")

else:
    print(f"The key '{key_to_edit}' was not found in the dictionary.")


