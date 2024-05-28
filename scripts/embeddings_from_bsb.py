#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:40:32 2024

@author: dgaio
"""

import os
import json
import openai
import numpy as np
import pickle
import csv
import time
from datetime import datetime



def get_embeddings(data_dict, verbose='no'):
    embeddings_dict = {}
    failed_samples = []
    sample_ids = list(data_dict.keys())
    descriptions = list(data_dict.values())

    try:
        start_api_call_time = time.time()
        batch_size = 20
        for i in range(0, len(descriptions), batch_size):
            chunk = descriptions[i:i + batch_size]
            sample_ids_chunk = sample_ids[i:i + batch_size]
            try:
                response = openai.Embedding.create(input=chunk, engine="text-embedding-ada-002")
                embeddings = [embedding['embedding'] for embedding in response['data']]
                for j, sample_id in enumerate(sample_ids_chunk):
                    embeddings_dict[sample_id] = np.array(embeddings[j], dtype=np.float32)
            except openai.APIError as e:
                print(f"API error for batch starting with {sample_ids_chunk[0]}: {e}")
                failed_samples.extend(sample_ids_chunk)
            except Exception as e:
                print(f"Unexpected error for batch starting with {sample_ids_chunk[0]}: {e}")
                failed_samples.extend(sample_ids_chunk)

        end_api_call_time = time.time()
        if verbose.lower() == 'yes':
            print(f"API call for {len(descriptions)} texts took {end_api_call_time - start_api_call_time:.2f} seconds")
    except Exception as e:
        print(f"General error in embedding function: {e}")

    return embeddings_dict, failed_samples





def load_data(source, source_type):
    if source_type == 'pickle':
        with open(source, 'rb') as file:
            return pickle.load(file)
    elif source_type == 'csv':
        with open(source, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            if 'col_4' not in reader.fieldnames:
                print(f"Warning: 'col_4' not found in: \n {source}. \nSkipping this file.\n")
                return None
            return {row['col_0']: f"{row['col_1']} - {row['col_4']}" for row in reader if 'col_4' in row}
    else:
        raise ValueError("Unsupported source type")



# =============================================================================
# def process_file(file_path, output_dir):
#     filename = os.path.basename(file_path)
#     embedding_filename = f"{filename.split('.')[0]}_bsbembeddings.json"
#     embedding_filepath = os.path.join(output_dir, embedding_filename)
# 
#     if os.path.exists(embedding_filepath):
#         print(f"Embedding file already exists: {embedding_filepath}")
#         return
# 
#     # Assuming the source type from the file extension
#     source_type = 'pickle' if file_path.endswith('gold_dict.pkl') else 'csv'
#     
#     data_dict = load_data(file_path, source_type)
#     embeddings, _ = get_embeddings(data_dict)
#     
#     with open(embedding_filepath, 'w') as f:
#         json.dump(embeddings, f)
#     print(f"Saved new embeddings to {embedding_filepath}")
# =============================================================================

def process_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    embedding_filename = f"{filename.split('.')[0]}_bsbembeddings.json"
    embedding_filepath = os.path.join(output_dir, embedding_filename)

    if os.path.exists(embedding_filepath):
        print(f"Embedding file already exists: {embedding_filepath}")
        return

    # Assuming the source type from the file extension
    source_type = 'pickle' if file_path.endswith('.pkl') else 'csv'
    data_dict = load_data(file_path, source_type)

    if data_dict is None:  # Check if data_dict is not empty
        #print(f"No valid data found in {file_path}. Skipping embedding generation.")
        return

    embeddings, failed_samples = get_embeddings(data_dict)

    with open(embedding_filepath, 'w') as f:
        json.dump(embeddings, f)
    print(f"Saved new embeddings to {embedding_filepath}")



def main():
    directory = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'
    output_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize API key
    with open('/Users/dgaio/my_api_key_embeddings', "r") as file:
        openai.api_key = file.read().strip()


    # List for gpt_clean_output files
    files_to_process = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('gpt_clean_output')]

    # Add specific path for gold_dict.pkl
    gold_dict_path = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"
    files_to_process.append(gold_dict_path)  # Add the gold_dict path to the list

    # for testing: 
    files_to_process = files_to_process[:100]
    
    for file_path in files_to_process:
        process_file(file_path, output_dir)
        
    print(len(files_to_process))



if __name__ == "__main__":
    main()











