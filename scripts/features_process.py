#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:24:23 2024

@author: dgaio
"""

import os
import pandas as pd
import csv
import pickle


# biomes

def find_distinguishing_features(files):
    """
    Determine the distinguishing features between filenames.
    Collect all tokens and identify those that are unique to some but not all filenames.
    """
    all_tokens = []
    file_tokens = []

    for file in files:
        tokens = os.path.basename(file).split('_')[:-1]  # to exclude date and time
        file_tokens.append(set(tokens))
        all_tokens.extend(tokens)

    token_count = {}
    for token in set(all_tokens):
        token_count[token] = sum(1 for tokens in file_tokens if token in tokens)

    # find tokens that are unique to some files but not common to all
    num_files = len(files)
    distinguishing_tokens = {token for token, count in token_count.items() if count != num_files}

    return distinguishing_tokens


def extract_labels_from_filename(filename, distinguishing_tokens):
    """
    Extract distinguishing labels from the filename based on the distinguishing tokens.
    """
    tokens = os.path.basename(filename).split('_')[:-1]  # to exclude date and time
    labels = [token for token in tokens if token in distinguishing_tokens]
    return ", ".join(labels)



def edit_features(file_label_map):
    """
    Allow the user to edit each label extracted from filenames, maintaining the dictionary structure.
    """
    print("Current labels for each file:")
    for idx, (file, label) in enumerate(file_label_map.items(), start=1):
        print(f"{idx}. {os.path.basename(file)} - {label}")

    if input("Do you want to edit any labels? (y/n): ").strip().lower() == 'y':
        for file in list(file_label_map.keys()):
            current_label = file_label_map[file]
            new_label = input(f"Change the label for '{os.path.basename(file)}' from '{current_label}' to (press enter to keep the same): ")
            if new_label:
                file_label_map[file] = new_label

    return file_label_map



def handle_malformed_lines(lines, filepath):
    
    with open(filepath, 'a') as f:
        for line in lines:
            f.write(line + '\n')  # write each malformed line to file

def load_and_process_file(file_name, gold_standard_df, label):
    
    mypath = os.path.dirname(file_name)
    malformed_file = os.path.join(mypath, 'malformed_lines.txt')
    
    # read and handle malformed lines
    dfr = pd.read_csv(file_name, header=None, engine='python', on_bad_lines=lambda lines: handle_malformed_lines(lines, malformed_file))
    
    # Proceed with your usual processing
    dfr = dfr.iloc[:, [0, 1]]
    dfr.columns = ['sample', 'gpt_biome']
    dfr['label'] = label

    merged_df = pd.merge(dfr, gold_standard_df, on='sample', how='inner')

    return merged_df





# vuoi veram che te lo spieghi vermanete? in realtaa e' abbstanza semplice...'



# sub-biomes

def filter_common_keys(embeddings_dict1, embeddings_dict2):
    common_keys = set(embeddings_dict1.keys()) & set(embeddings_dict2.keys())
    filtered_dict1 = {k: embeddings_dict1[k] for k in common_keys}
    filtered_dict2 = {k: embeddings_dict2[k] for k in common_keys}
    return filtered_dict1, filtered_dict2



