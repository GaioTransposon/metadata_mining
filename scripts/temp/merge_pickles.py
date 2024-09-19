#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:45:13 2024

@author: dgaio
"""

# merging of pkl files: 

import pickle
import glob
import os

def merge_pkl_files(input_dir, output_file):
    combined_data = {}

    # Iterate over all .pkl files in the input directory
    for file_path in glob.glob(os.path.join(input_dir, '*.pkl')):
        # Load the current .pkl file
        with open(file_path, 'rb') as current_pkl:
            data = pickle.load(current_pkl)
            # Update the combined_data dictionary, checking for duplicates
            for key, value in data.items():
                if key in combined_data:
                    print(f"Duplicate key {key} found in {file_path}. Skipping.")
                    continue
                combined_data[key] = value

    # Save the combined data into a new .pkl file
    with open(output_file, 'wb') as output_pkl:
        pickle.dump(combined_data, output_pkl)

    print(f"All .pkl files in {input_dir} have been merged into {output_file}.")

# Usage
input_directory = "/mnt/mnemo5/dgaio/MicrobeAtlasProject/temp"  # Change to your directory path
output_pkl_file = input_directory + "/combined_data.pkl"  # Name for the combined .pkl file
merge_pkl_files(input_directory, output_pkl_file)
