#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:16:55 2024

@author: dgaio
"""

import os
import pickle

def load_gold_dict(gold_dict_path):
    if not os.path.exists(gold_dict_path):
        print(f"Gold dict file not found: {gold_dict_path}")
        return {}
    with open(gold_dict_path, 'rb') as file:
        gold_dict = pickle.load(file)
        gold_dict = gold_dict[0]
    print(f"Loaded {len(gold_dict)} entries from the gold dictionary.")
    return gold_dict if isinstance(gold_dict, list) else gold_dict

def find_files_with_gold_samples(temp_dir, gold_dict):
    if not os.path.exists(temp_dir):
        print(f"Temp directory not found: {temp_dir}")
        return []
    gold_sample_files = []
    for filename in os.listdir(temp_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'rb') as file:
                batch_data = pickle.load(file)
                if any(sample_id in gold_dict for sample_id in batch_data.keys()):
                    gold_sample_files.append(filename)
    return gold_sample_files

def main():
    work_dir = "/mnt/mnemo5/dgaio/MicrobeAtlasProject/"
    temp_dir = os.path.join(work_dir, 'temp')
    gold_dict_path = os.path.join(work_dir, 'gold_dict.pkl')

    gold_dict = load_gold_dict(gold_dict_path)
    if not gold_dict:
        return

    gold_sample_files = find_files_with_gold_samples(temp_dir, gold_dict)
    if not gold_sample_files:
        print("No files found containing gold dict samples.")
        return

    print("Files containing gold dict samples:")
    for filename in gold_sample_files:
        print(filename)

if __name__ == "__main__":
    main()
