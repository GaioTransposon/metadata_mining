#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:40:32 2024

@author: dgaio
"""

# makes a check if json already exists it won t re do it 

import csv
import openai
import json
import os
import pickle
import re

# Set directory and API key paths
directory_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'
output_dir = os.path.join(directory_path, 'embeddings')
api_key_path = '/Users/dgaio/my_api_key_embeddings'
gold_dict_path = '/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl'

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
                print(values[2])
                sub_biome_text = values[2]
                biome = values[1] if len(values) > 1 else 'unknown'
                gold_dict_sb[key] = {
                    'sub-biome': sub_biome_text,
                    'biome': biome
                }
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
            response = openai.Embedding.create(input=chunk, engine="text-embedding-3-small") # text-embedding-ada-002 # text-embedding-3-small # text-embedding-3-large
            embeddings = [embedding['embedding'] for embedding in response['data']]
            for j, sample_id in enumerate(sample_ids_chunk):
                embeddings_dict[sample_id] = {
                    'embedding': embeddings[j],
                    'sub-biome': data_dict[sample_id]['sub-biome']
                }
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
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 5:
                continue
            sample_id = row[0].strip()
            sub_biome_text = re.sub(r'\{.*?\}', '', row[4])
            sub_biome_text = re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', sub_biome_text).strip()
            if not sub_biome_text or sub_biome_text.lower() in ['na', 'unknown', ''] or not re.search(r'[a-zA-Z]', sub_biome_text):
                continue
            samples[sample_id] = {'sub-biome': sub_biome_text}

            # Debugging output
            print("##############")
            print(sample_id)
            print(sub_biome_text)
            print("##############")

    embeddings_dict, failed_samples = get_embeddings(samples, include_biome=False)
    base_filename = os.path.basename(csv_file_path)
    output_filename = base_filename.replace('.csv', '_sbembeddings.json').replace('.txt', '_sbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)

    # Assuming embeddings_dict is serializable
    if embeddings_dict:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_dict, f, ensure_ascii=False, indent=4)

    return embeddings_dict, failed_samples, output_file_path





# Save embeddings to a JSON file
def save_embeddings(embeddings_dict, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(embeddings_dict, json_file, ensure_ascii=False, indent=4)


###
# Process the gold dictionary
output_file_path = os.path.join(output_dir, 'gold_dict_sbembeddings.json')
# Check if the embeddings file already exists
if os.path.exists(output_file_path):
    print('Embeddings file already exists:', output_file_path)
else:
    gold_dict_sb = load_and_extract_sub_biome(gold_dict_path)
    embeddings_dict, failed_samples = get_embeddings(gold_dict_sb, include_biome=True)
    save_embeddings(embeddings_dict, output_file_path)
    print('Embeddings saved to:', output_file_path)
###
    
    

my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp1.0_presp1.5_rs22_API132_normal_dt202406051450.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']




# Check and process each file
for filename in my_files:
    file_path = os.path.join(directory_path, filename)
    output_filename = filename.replace('.csv', '_sbembeddings.json').replace('.txt', '_sbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_file_path):
        print('Embeddings already exist for:', output_file_path)
        continue

    print('\nGetting embeddings for:', file_path)
    embeddings_dict, failed_samples, output_file_path = process_file(file_path, output_dir)
    save_embeddings(embeddings_dict, output_file_path)
    print('Embeddings saved to:', output_file_path)
    











    

    