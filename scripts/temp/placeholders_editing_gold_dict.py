#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:29:26 2024

@author: dgaio
"""

import pandas as pd
import pickle


# Load the .pkl file
data = pd.read_pickle('/Users/dgaio/github/metadata_mining/source_data/gold_dict_backup.pkl')
print(type(data))

data_dict_ori = data[0]  # This assumes the first element of the tuple is your dictionary
data_dict_ori_pmids = data[1] 

data_dict = data_dict_ori




# Define the placeholder value to be inserted
placeholder = 'placeholder'

# Iterate over the dictionary
for key, value in data_dict.items():
    # Check if the tuple contains the string 'geo_text'
    if any('geo:' in str(item) for item in value):
        # Find the index of the element containing 'geo_text'
        index = next(i for i, item in enumerate(value) if 'geo:' in str(item))
        print(index)
        # Convert the tuple to list to allow modifications
        value_list = list(value)
        print(value_list)
        
        
        # Insert the placeholder before the 'geo_text' element
        value_list.insert(index, placeholder)
        print(value_list)
        
        # Update the dictionary with the modified list converted back to tuple
        data_dict[key] = tuple(value_list)

# Print the updated dictionary to verify changes
print(data_dict)



# Iterate over the dictionary items
for key, value in data_dict.items():
    # Process each item in the tuple
    new_value = []
    for item in value:
        if isinstance(item, str):  # Check if the item is a string
            # Check and perform replacements if needed
            new_item = item.replace('geo: ', '').replace('geo_text: ', '')
            if item != new_item:  # If a replacement has occurred
                print(f"Original: {item}, Replaced: {new_item}")  # Print the changes
            new_value.append(new_item)
        else:
            new_value.append(item)
    # Update the dictionary entry with the new tuple
    data_dict[key] = tuple(new_value)

# Print the updated dictionary to verify changes
print(data_dict)






# Counters
total_keys = len(data_dict)
biome_count = {}
more_than_two_values_count = 0
placeholder_count = 0
placeholder_in_biome_count = {}
more_than_two_values_no_placeholder = {}

# Iterate over the dictionary
for key, value in data_dict.items():
    # Count keys per unique biome
    biome = value[1]
    biome_count[biome] = biome_count.get(biome, 0) + 1

    # Check if tuple has more than two values
    if len(value) > 2:
        more_than_two_values_count += 1
        has_placeholder = 'placeholder' in value
        # Count keys with 'placeholder'
        if has_placeholder:
            placeholder_count += 1
            placeholder_in_biome_count[biome] = placeholder_in_biome_count.get(biome, 0) + 1
        else:
            # Count per biome keys with more than two values and no 'placeholder'
            more_than_two_values_no_placeholder[biome] = more_than_two_values_no_placeholder.get(biome, 0) + 1

# Print results
print("Total number of keys:", total_keys)
print("Number of keys per unique biome:", biome_count)
print("Number of keys with more than two values:", more_than_two_values_count)
print("Number of keys with 'placeholder' among their values:", placeholder_count)
print("Number of keys per unique biome with 'placeholder' among their values:", placeholder_in_biome_count)
print("Number of keys per biome, with more than two values, that do NOT have 'placeholder':", more_than_two_values_no_placeholder)






# Combine the updated dictionary with PMIDs into a tuple
updated_data = (data_dict, data_dict_ori_pmids)

# Specify the path where you want to save the file
file_path = '/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl'

# Open a file for writing
with open(file_path, 'wb') as file:
    # Use pickle.dump to serialize the tuple and write it to the file
    pickle.dump(updated_data, file)
    
    
    



