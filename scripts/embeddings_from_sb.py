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

# Set directory and api key paths
directory_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'
output_dir = os.path.join(directory_path, 'embeddings')
api_key_path = '/Users/dgaio/my_api_key_embeddings'
gold_dict_path = '/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl'

# Load and combine ground truth for each sample
def load_and_combine_gold_dict(gold_dict_path):
    gold_dict_sb = {}
    with open(gold_dict_path, 'rb') as file:
        data = pickle.load(file)
        for key, values in data.items():
            if len(values) >= 3:
                combined_text = f"{values[1]} - {values[2]}"
                gold_dict_sb[key] = combined_text
    return gold_dict_sb

# Retrieve embeddings in batches
def get_embeddings(data_dict):
    embeddings_dict = {}
    failed_samples = []
    sample_ids = list(data_dict.keys())
    descriptions = list(data_dict.values())
    batch_size = 20

    for i in range(0, len(descriptions), batch_size):
        chunk = descriptions[i:i + batch_size]
        sample_ids_chunk = sample_ids[i:i + batch_size]
        try:
            response = openai.Embedding.create(input=chunk, engine="text-embedding-ada-002")
            embeddings = [embedding['embedding'] for embedding in response['data']]
            for j, sample_id in enumerate(sample_ids_chunk):
                embeddings_dict[sample_id] = embeddings[j]
        except Exception as e:
            print(f"Failed to retrieve embeddings for {sample_ids_chunk[0]} to {sample_ids_chunk[-1]}: {e}")
            failed_samples.extend(sample_ids_chunk)

    return embeddings_dict, failed_samples

# Process each input file, retrieve embeddings for biome and sub-biome
def process_file(csv_file_path):
    samples = {}
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header 
        for row in reader:
            if len(row) < 5:
                continue  
            sample_id = row[0]
            combined_text = f"{row[1]} - {row[4]}"
            samples[sample_id] = combined_text

    embeddings_dict, failed_samples = get_embeddings(samples)
    base_filename = os.path.basename(csv_file_path)
    output_filename = base_filename.replace('.csv', '_sbembeddings.json').replace('.txt', '_sbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)

    return embeddings_dict, failed_samples, output_file_path

# Save embeddings to a JSON file
def save_embeddings(embeddings_dict, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(embeddings_dict, json_file)

# Set OpenAI API key
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()



input_filenames = my_files   # my_files is in middle_dir/my_files.txt


# Check and process each file
for filename in input_filenames:
    file_path = os.path.join(directory_path, filename)
    base_filename = os.path.basename(file_path)
    output_filename = base_filename.replace('.csv', '_sbembeddings.json').replace('.txt', '_sbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_file_path):
        print('Embeddings already exist for:', output_file_path)
        continue

    print('\nGetting embeddings for:', file_path)
    embeddings_dict, failed_samples, output_file_path = process_file(file_path)
    save_embeddings(embeddings_dict, output_file_path)
    print('Embeddings saved to:', output_file_path)
    
    
    
    
    



    

    