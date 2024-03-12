#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:49:46 2024

@author: dgaio
"""

import numpy as np
import pickle
import os



def check_embeddings_in_batch(pkl_file_path, batch_size=100000):
    not_numpy_array_samples = []
    unexpected_shape_samples = []
    expected_shape_count = 0  
    total_samples_checked = 0

    with open(pkl_file_path, 'rb') as file:
        while True:
            try:
                batch = pickle.load(file)
                for sample_id, embedding in batch.items():
       
                    if not isinstance(embedding, np.ndarray):
                        not_numpy_array_samples.append(sample_id)
                    elif embedding.shape != (1536,):  # expects a 1D array of length 1536
                        unexpected_shape_samples.append(sample_id)
                    else:
                        expected_shape_count += 1  
                         
                    total_samples_checked += 1

                    if total_samples_checked % batch_size == 0:
                        print(f"Checked {total_samples_checked} samples so far...")
            except EOFError:
                break  # end of file reached

    return not_numpy_array_samples, unexpected_shape_samples, expected_shape_count



def list_expected_samples(work_dir, metadata_directory):
    expected_samples = set()
    for subdir in os.listdir(os.path.join(work_dir, metadata_directory)):
        subdir_path = os.path.join(work_dir, metadata_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        sample_files = [f for f in os.listdir(subdir_path) if f.endswith('_clean.txt')]
        for sample_file in sample_files:
            sample_id = sample_file.split('_clean')[0]
            expected_samples.add(sample_id)
    return expected_samples



def check_missing_samples(combined_pkl_file, expected_samples):
    data_samples = set()
    with open(combined_pkl_file, 'rb') as file:
        while True:
            try:
                batch = pickle.load(file)
                data_samples.update(batch.keys())
            except EOFError:
                break  # end of file reached

    missing_samples = expected_samples - data_samples
    return missing_samples



work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"  
metadata_directory = "sample.info_split_dirs/"  
combined_pkl_file = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/temp/combined_data.pkl"  

# check if all embeddings look fine in batches
not_numpy_array_samples, unexpected_shape_samples, expected_shape_count = check_embeddings_in_batch(combined_pkl_file)

# list all expected samples based on subdirectories and metadata files
expected_samples = list_expected_samples(work_dir, metadata_directory)

# check if embeddings for all expected samples are present
missing_samples = check_missing_samples(combined_pkl_file, expected_samples)

print(f"Total samples with expected shape: {expected_shape_count}")
print(f"Total non-NumPy array samples: {len(not_numpy_array_samples)}. Examples: {not_numpy_array_samples[:5]}")
print(f"Total samples with unexpected shape: {len(unexpected_shape_samples)}. Examples: {unexpected_shape_samples[:5]}")
print(f"Total missing samples: {len(missing_samples)}. Examples: {list(missing_samples)[:5]}")






