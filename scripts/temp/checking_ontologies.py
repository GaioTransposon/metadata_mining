#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:13:34 2023

@author: dgaio
"""

import pickle

def search_partial_key(pkl_file, partial_key):
    try:
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
            matching_values = {}

            for key in data:
                if partial_key in key:
                    matching_values[key] = data[key]

            return matching_values
    except FileNotFoundError:
        return None

# Usage example:
pkl_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/ontologies_dict.pkl'  # Replace with the path to your pickle file
partial_key = 'ENVO_00001998'  # Replace with the partial key you want to search for

result = search_partial_key(pkl_file, partial_key)
if result:
    for key, value in result.items():
        print(f"Key: {key}, Value: {value}")
else:
    print(f"No matching values found for partial key: {partial_key}")




import pickle
import re
from collections import defaultdict

# Function to extract digits from a string
def extract_digits(s):
    return ''.join(filter(str.isdigit, s))

# Function to remove digits from a string
def remove_digits(s):
    return re.sub(r'\d+', '', s)

# Load your pickle file
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Group keys by their digit parts
groups = defaultdict(list)
for key in data.keys():
    digit_part = extract_digits(key)
    groups[digit_part].append(key)

# Find groups with same digits but different non-digit characters
result = {}
for digit, keys in groups.items():
    if len(set(map(remove_digits, keys))) > 1:
        result[digit] = keys

print("Groups with same digits but different non-digit characters:", result)


