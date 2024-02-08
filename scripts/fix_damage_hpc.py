#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:16:55 2024

@author: dgaio
"""

import os
import pickle

def get_sample_ids_from_files(temp_dir, target_date):
    sample_ids = set()  # Using a set to avoid duplicates
    for filename in os.listdir(temp_dir):
        if filename.startswith("embeddings_batch_") and target_date in filename:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'rb') as file:
                batch_data = pickle.load(file)
                sample_ids.update(batch_data.keys())
    return sample_ids

def main():
    work_dir = "/mnt/mnemo5/dgaio/MicrobeAtlasProject/"
    temp_dir = os.path.join(work_dir, 'temp')
    target_date = "20240207"  # Format: YYYYMMDD

    sample_ids = get_sample_ids_from_files(temp_dir, target_date)

    # Write the sample IDs to to_del.txt
    with open(os.path.join(work_dir, 'to_del.txt'), 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")

if __name__ == "__main__":
    main()


