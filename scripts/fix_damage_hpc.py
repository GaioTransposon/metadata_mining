#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:16:55 2024

@author: dgaio
"""

import os
import pickle

def load_gold_dict(gold_dict_path):
    with open(gold_dict_path, 'rb') as file:
        gold_dict = pickle.load(file)
    # Assuming gold_dict is a dictionary with sample IDs as keys
    return gold_dict[0] if isinstance(gold_dict, list) else gold_dict

def find_files_with_gold_samples(temp_dir, gold_dict):
    gold_sample_files = []
    for filename in os.listdir(temp_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'rb') as file:
                batch_data = pickle.load(file)
                # Check if any sample ID in the current file is in the gold_dict
                if any(sample_id in gold_dict for sample_id in batch_data.keys()):
                    gold_sample_files.append(filename)
    return gold_sample_files

def main():
    work_dir = "/mnt/mnemo5/dgaio/MicrobeAtlasProject/"
    temp_dir = os.path.join(work_dir, 'temp')
    gold_dict_path = os.path.join(work_dir, 'gold_dict.pkl')

    gold_dict = load_gold_dict(gold_dict_path)
    gold_sample_files = find_files_with_gold_samples(temp_dir, gold_dict)

    print("Files containing gold dict samples:")
    for filename in gold_sample_files:
        print(filename)

if __name__ == "__main__":
    main()
