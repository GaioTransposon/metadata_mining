#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:51:32 2024

@author: dgaio
"""


import os
import openai
import numpy as np
import time
import json
import tempfile

# Constants
RETRY_LIMIT = 3
CHECKPOINT_FILE = 'embeddings_checkpoint.json'
METADATA_DIRECTORY = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"
TOTAL_DIRS = 1000  # Assuming directories go from 000 to 999
PRINT_EVERY = 10  # Print status every 100 samples

# open api key
api_key_path ='/Users/dgaio/my_api_key'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()
    
    
# Updated functions
def get_embedding(text):
    retries = 0
    while retries < RETRY_LIMIT:
        try:
            response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error: {e}, Retrying {retries + 1}/{RETRY_LIMIT}")
            retries += 1
            time.sleep(1)  # Sleep for a brief moment before retrying
    return None


def save_checkpoint(embeddings, checkpoint_file):
    temp_fd, temp_path = tempfile.mkstemp()
    try:
        with os.fdopen(temp_fd, 'w') as temp_file:
            json.dump(embeddings, temp_file)
        os.replace(temp_path, checkpoint_file)
    except Exception as e:
        print(f"Error while saving checkpoint: {e}")
        os.remove(temp_path)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            return json.load(file)
    return {}

def process_directory(dir_name, processed_count, start_time):
    dir_path = os.path.join(METADATA_DIRECTORY, dir_name)
    for file_name in os.listdir(dir_path):
        if file_name.endswith('_clean.txt'):
            sample_id = file_name.replace('_clean.txt', '')
            if sample_id not in checkpoint_data:
                metadata_file_path = os.path.join(dir_path, file_name)
                with open(metadata_file_path, 'r') as file:
                    metadata = file.read()
                #print(sample_id)
                embedding = get_embedding(metadata)
                #print(len(embedding))
                if embedding is not None:
                    checkpoint_data[sample_id] = embedding.tolist()
                    save_checkpoint(checkpoint_data, CHECKPOINT_FILE)
                    processed_count += 1
                    if processed_count % PRINT_EVERY == 0:
                        elapsed_time = time.time() - start_time
                        average_time_per_sample = elapsed_time / processed_count
                        print(f"Processed {processed_count} samples so far. "
                              f"Elapsed time: {elapsed_time:.2f} seconds. "
                              f"Average time per sample: {average_time_per_sample:.4f} seconds/sample")
    return processed_count


# Main Script
checkpoint_data = load_checkpoint(CHECKPOINT_FILE)
processed_samples = 0
start_time = time.time()

for i in range(TOTAL_DIRS):
    dir_name = f"dir_{i:03d}"
    processed_samples = process_directory(dir_name, processed_samples, start_time)

elapsed_time = time.time() - start_time
average_time_per_sample = elapsed_time / processed_samples if processed_samples > 0 else 0
print(f"Total samples processed: {processed_samples}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")
print(f"Average time per sample: {average_time_per_sample:.4f} seconds/sample")
