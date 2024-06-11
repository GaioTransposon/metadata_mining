#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:40:32 2024

@author: dgaio
"""

import csv
import openai
import json
import glob
import os
import pickle


directory_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'
output_dir = os.path.join(directory_path, 'embeddings')
api_key_path = '/Users/dgaio/my_api_key_embeddings'
gold_dict_path = '/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl'
input_file_pattern = 'gpt_clean_output*.csv'


# loads ground truth and combines for each sample, biome with sub-biome
def load_and_combine_gold_dict(gold_dict_path):
    gold_dict_bsb = {}
    with open(gold_dict_path, 'rb') as file:
        data = pickle.load(file)
        for key, values in data.items():
            if len(values) >= 3:
                combined_text = f"{values[1]} - {values[2]}"
                #print(key)  
                #print(combined_text)  
                gold_dict_bsb[key] = combined_text
    return gold_dict_bsb


# get embeddings in batches (it's faster)
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


# opens gpt output files, gets biome and sub-biome, and gets embeddings for those (for each sample)
def process_file(csv_file_path):
    samples = {}
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header 
        for row in reader:
            # skip files taht dont have at leats 5 columns
            if len(row) < 5:
                continue  
            sample_id = row[0]
            combined_text = f"{row[1]} - {row[4]}"
            samples[sample_id] = combined_text

    embeddings_dict, failed_samples = get_embeddings(samples)

    # output filename based on input filename
    base_filename = os.path.basename(csv_file_path)
    output_filename = base_filename.replace('.csv', '_bsbembeddings.json')
    output_file_path = os.path.join(output_dir, output_filename)

    return embeddings_dict, failed_samples, output_file_path



def save_embeddings(embeddings_dict, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(embeddings_dict, json_file)


        

with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()




############
# Get embeddings from gpt output files: 
file_paths = glob.glob(os.path.join(directory_path, input_file_pattern))
existing_output_files = set(os.path.basename(f) for f in glob.glob(os.path.join(output_dir, '*_bsbembeddings.json')))
# processed_files = 0

for file_path in file_paths:
    base_filename = os.path.basename(file_path)
    expected_output_filename = base_filename.replace('.csv', '_bsbembeddings.json')
    
    if expected_output_filename in existing_output_files:
        
        print('File:\n', expected_output_filename, '\nexists')
        continue  

    # for testing purposes: 
    # if processed_files >= 2:
    #     break
    
    print('\nGetting embeddings for:\n', file_path)
    process_file(file_path)
    # processed_files += 1

# print(f"Processed {processed_files} files.")
############



############
# Get embeddings from ground truth: 
gold_dict_bsb = load_and_combine_gold_dict(gold_dict_path)
output_filename = 'gold_dict_bsbembeddings.json'
output_file_path = os.path.join(output_dir, output_filename)

if os.path.exists(output_file_path):
    print('Gold dict embeddings file already exists:', output_file_path)
else:
    embeddings_dict, failed_samples = get_embeddings(gold_dict_bsb)
    print('Failed samples for file: ', output_file_path)
    print(failed_samples)
    
    save_embeddings(embeddings_dict, output_file_path)
    
    print(f"Embeddings for gold dict saved to {output_file_path}.")
############








    