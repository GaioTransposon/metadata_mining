#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:40:32 2024

@author: dgaio
"""


# Runs for all .txt or .csv files
# if json already exists it won t re get the embeddings for it

# NB before running, run: 
# pip uninstall
# pip install openai==0.28
# restart spyder 


import os
import csv
import openai
import json

import pickle
import re
import glob

# Set directory and API key paths
directory_path = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject'   # UZH: /Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject
output_dir = os.path.join(directory_path, 'embeddings')
api_key_path = '/Users/danielagaio/Desktop/keys/my_api_key_embeddings'    # UZH: /Users/dgaio/my_api_key_embeddings
gold_dict_path = '/Users/danielagaio/github/metadata_mining/source_data/gold_dict.pkl'   # UZH: /Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl

# Set OpenAI API key
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()

# Load and extract sub-biome and biome for each sample
def load_and_extract_sub_biome(gold_dict_path):
    gold_dict_sb = {}
    with open(gold_dict_path, 'rb') as file:
        data = pickle.load(file)
        for key, values in data.items():
            if len(values) >= 3:
                sub_biome_text = values[2]
                biome = values[1] if len(values) > 1 else 'unknown'
                gold_dict_sb[key] = {'sub-biome': sub_biome_text, 'biome': biome}
    return gold_dict_sb

# Retrieve embeddings in batches
def get_embeddings(data_dict, include_biome=False):
    embeddings_dict = {}
    failed_samples = []
    sample_ids = list(data_dict.keys())
    descriptions = [value['sub-biome'] for value in data_dict.values()]
    batch_size = 20

    for i in range(0, len(descriptions), batch_size):
        chunk = descriptions[i:i + batch_size]
        sample_ids_chunk = sample_ids[i:i + batch_size]
        try:
            response = openai.Embedding.create(input=chunk, engine="text-embedding-3-small")
            embeddings = [embedding['embedding'] for embedding in response['data']]
            for j, sample_id in enumerate(sample_ids_chunk):
                embeddings_dict[sample_id] = {'embedding': embeddings[j], 'sub-biome': data_dict[sample_id]['sub-biome']}
                if include_biome:
                    embeddings_dict[sample_id]['biome'] = data_dict[sample_id]['biome']
        except Exception as e:
            print(f"Failed to retrieve embeddings for {sample_ids_chunk[0]} to {sample_ids_chunk[-1]}: {e}")
            failed_samples.extend(sample_ids_chunk)

    return embeddings_dict, failed_samples

# Process each input file, retrieve embeddings for sub-biome
def process_file(csv_file_path, output_dir):
    samples = {}
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 5:
                continue
            sample_id = row[0].strip()
            sub_biome_text = re.sub(r'\{.*?\}', '', row[4])
            sub_biome_text = re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', sub_biome_text).strip()
            if not sub_biome_text or sub_biome_text.lower() in ['na', 'unknown', ''] or not re.search(r'[a-zA-Z]', sub_biome_text):
                continue
            samples[sample_id] = {'sub-biome': sub_biome_text}

    embeddings_dict, failed_samples = get_embeddings(samples, include_biome=False)
    base_filename = os.path.basename(csv_file_path)
    output_filename = base_filename.replace('.csv', '_sbembeddings.json').replace('.txt', '_sbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)

    if embeddings_dict:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_dict, f, ensure_ascii=False, indent=4)

    return embeddings_dict, failed_samples, output_file_path

# Save embeddings to a JSON file
def save_embeddings(embeddings_dict, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(embeddings_dict, json_file, ensure_ascii=False, indent=4)

# Process the gold dictionary
output_file_path = os.path.join(output_dir, 'gold_dict_sbembeddings.json')
if not os.path.exists(output_file_path):
    gold_dict_sb = load_and_extract_sub_biome(gold_dict_path)
    embeddings_dict, failed_samples = get_embeddings(gold_dict_sb, include_biome=True)
    save_embeddings(embeddings_dict, output_file_path)
    print('Embeddings saved to:', output_file_path)

# Check and process each file that matches the pattern
pattern = os.path.join(directory_path, 'gpt_clean_output*')    # when running for George (2nd curator) file: GH_collect*
file_list = glob.glob(pattern + '.txt') + glob.glob(pattern + '.csv')
for file_path in file_list:
    output_filename = os.path.basename(file_path).replace('.csv', '_sbembeddings.json').replace('.txt', '_sbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_file_path):
        print('Embeddings already exist for:', output_file_path)
        continue

    print('\nGetting embeddings for:', file_path)
    embeddings_dict, failed_samples, output_file_path = process_file(file_path, output_dir)
    save_embeddings(embeddings_dict, output_file_path)
    print('Embeddings saved to:', output_file_path)



    

    