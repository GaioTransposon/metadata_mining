#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:51:32 2024

@author: dgaio
"""
import argparse
import os
import openai
import numpy as np
import time
import json
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Existing arguments unchanged
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--metadata_directory', type=str, required=True, help='Directory with split metadata')
    parser.add_argument('--batch_size', type=int, required=True, help='Number of samples to process at a time')
    parser.add_argument('--api_key_path', type=str, required=True, help='Path to api key')
    parser.add_argument('--processed_samples_file', type=str, required=True, help='File to track processed sample IDs')

    # New verbose argument
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')

    return parser.parse_args()


def load_processed_samples(processed_samples_file):
    if not os.path.exists(processed_samples_file):
        return set()
    with open(processed_samples_file, 'r') as file:
        return set(file.read().splitlines())


def append_processed_samples(processed_samples_file, sample_ids):
    with open(processed_samples_file, 'a') as file:
        for sample_id in sample_ids:
            file.write(sample_id + '\n')


def get_embeddings(texts, verbose):
    start_api_call_time = time.time()
    # Simulate API call with a placeholder function
    embeddings = np.random.rand(len(texts), 512)  # Placeholder for actual embeddings
    end_api_call_time = time.time()
    if verbose:
        print(f"{datetime.now()} - API call for {len(texts)} texts took {end_api_call_time - start_api_call_time:.2f} seconds")
    return embeddings, []





def save_embeddings_batch(embeddings, sample_ids, temp_dir, verbose):
    start_write_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")  # Including microseconds
    temp_filename = os.path.join(temp_dir, f"embeddings_batch_{timestamp}.json")

    batch_data = {sample_id: embedding.tolist() for sample_id, embedding in zip(sample_ids, embeddings)}
    with open(temp_filename, 'w') as file:
        json.dump(batch_data, file)

    end_write_time = time.time()
    if verbose:
        print(f"{datetime.now()} - Time taken to write and update embeddings: {end_write_time - start_write_time:.2f} seconds")
        print(f"{datetime.now()} - Saved batch to {temp_filename}")



def main():
    args = parse_arguments()

    # Initialize API key
    with open(args.api_key_path, "r") as file:
        openai.api_key = file.read().strip()

    processed_samples = load_processed_samples(os.path.join(args.work_dir, args.processed_samples_file))
    temp_dir = os.path.join(args.work_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)  # Ensure temp directory exists

    total_samples_processed_in_run = 0
    for subdir in os.listdir(os.path.join(args.work_dir, args.metadata_directory)):
        subdir_path = os.path.join(args.work_dir, args.metadata_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue

        sample_files = [f for f in os.listdir(subdir_path) if f.endswith('_clean.txt')]
        batch = []

        for sample_file in sample_files:
            sample_id = sample_file.split('_clean')[0]

            if sample_id in processed_samples:
                continue

            metadata_file_path = os.path.join(subdir_path, sample_file)
            with open(metadata_file_path, 'r') as file:
                metadata = file.read()

            batch.append((sample_id, metadata))

            if len(batch) >= args.batch_size:
                start_batch_time = time.time()
                sample_ids, metadata_texts = zip(*batch)
                metadata_embeddings, _ = get_embeddings(metadata_texts, args.verbose)
                save_embeddings_batch(metadata_embeddings, sample_ids, temp_dir, args.verbose)
                append_processed_samples(os.path.join(args.work_dir, args.processed_samples_file), sample_ids)
                total_samples_processed_in_run += len(batch)
                end_batch_time = time.time()
                print(f"{datetime.now()} - Processed batch of {len(batch)} samples in {end_batch_time - start_batch_time:.2f} seconds, n={total_samples_processed_in_run} samples processed in run, {(end_batch_time - start_batch_time) / len(batch):.2f} sec/sample.")
                batch = []

        # Handle the last batch if it exists
        if batch:
            start_batch_time = time.time()
            sample_ids, metadata_texts = zip(*batch)
            metadata_embeddings, _ = get_embeddings(metadata_texts, args.verbose)
            save_embeddings_batch(metadata_embeddings, sample_ids, temp_dir, args.verbose)
            append_processed_samples(os.path.join(args.work_dir, args.processed_samples_file), sample_ids)
            total_samples_processed_in_run += len(batch)
            end_batch_time = time.time()
            print(f"{datetime.now()} - Processed last batch of {len(batch)} samples in {end_batch_time - start_batch_time:.2f} seconds, n={total_samples_processed_in_run} samples processed in run, {(end_batch_time - start_batch_time) / len(batch):.2f} sec/sample.")

    print(f"{datetime.now()} - All samples processed.")


if __name__ == "__main__":
    main()

        



# python /Users/dgaio/github/metadata_mining/scripts/embeddings_from_metadata.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --metadata_directory "sample.info_split_dirs/" \
#     --batch_size 10 \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --processed_samples_file "processed_samples_file.txt" \
#     --verbose 


# ssh dgaio@phobos.mls.uzh.ch
# python /mnt/mnemo5/dgaio/github/metadata_mining/scripts/embeddings_from_metadata.py \
#     --work_dir "/mnt/mnemo5/dgaio/MicrobeAtlasProject/" \
#     --metadata_directory "sample.info_split_dirs/" \
#     --batch_size 10 \
#     --api_key_path "/mnt/mnemo5/dgaio/my_api_key" \
#     --processed_samples_file "processed_samples_file.txt" \
#     --verbose 
























# =============================================================================
# # 
# # To run tests in console: 
# #
# 
# import os
# import openai
# import numpy as np
# import time
# import json
# import shutil
# 
# # Constants
# METADATA_DIRECTORY = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs_test"
# EMBEDDINGS_FILE = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings.json"
# TEMP_EMBEDDINGS_FILE = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/temp_embeddings.json"
# BATCH_SIZE = 10  # Adjust based on token limit estimation
# 
# # Load API key
# api_key_path = '/Users/dgaio/my_api_key'
# with open(api_key_path, "r") as file:
#     openai.api_key = file.read().strip()
# 
# 
# def get_embeddings(texts):
#     try:
#         response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
#         embeddings = [embedding['embedding'] for embedding in response['data']]
#         return np.array(embeddings), []
#     except Exception as e:
#         print(f"Error in batch embedding: {e}")
#         # Return empty embeddings and the full list of texts as failed samples
#         return np.array([]), texts
#     
# 
# def save_embeddings(embeddings, sample_ids):
#     if not os.path.exists(EMBEDDINGS_FILE):
#         with open(EMBEDDINGS_FILE, 'w') as file:
#             json.dump({}, file)  # Create an empty JSON file if it doesn't exist
# 
#     with open(EMBEDDINGS_FILE, 'r') as file:
#         data = json.load(file)
#         for sample_id, embedding in zip(sample_ids, embeddings):
#             data[sample_id] = embedding.tolist()
# 
#     with open(TEMP_EMBEDDINGS_FILE, 'w') as temp_file:
#         json.dump(data, temp_file)
# 
#     # Replace the original file with the updated temporary file
#     shutil.move(TEMP_EMBEDDINGS_FILE, EMBEDDINGS_FILE)
# 
# def load_processed_samples():
#     if not os.path.exists(EMBEDDINGS_FILE):
#         return set()
#     with open(EMBEDDINGS_FILE, 'r') as file:
#         data = json.load(file)
#     return set(data.keys())
# 
# # Main Processing
# processed_samples = load_processed_samples()
# failed_samples = []
# total_samples_processed = 0  # Initialize the count
# 
# for subdir in os.listdir(METADATA_DIRECTORY):
#     subdir_path = os.path.join(METADATA_DIRECTORY, subdir)
#     if not os.path.isdir(subdir_path):
#         continue
# 
#     sample_files = [f for f in os.listdir(subdir_path) if f.endswith('_clean.txt')]
#     batch = []
#     
#     for sample_file in sample_files:
#         sample_id = sample_file.split('_clean')[0]
#         if sample_id in processed_samples:
#             continue
# 
#         metadata_file_path = os.path.join(subdir_path, sample_file)
#         with open(metadata_file_path, 'r') as file:
#             metadata = file.read()
#         
#         batch.append((sample_id, metadata))
#         
#         if len(batch) >= BATCH_SIZE:
#             start_time = time.time()
#             sample_ids, metadata_texts = zip(*batch)
#             metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
#             failed_samples.extend(batch_failed_samples)
#             end_time = time.time()
# 
#             save_embeddings(metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
#             time_per_sample = (end_time - start_time) / len(batch)
#             total_samples_processed += len(batch)  # Increment the count
#             print(f"Processed batch of {len(batch)} samples in {end_time - start_time:.2f} seconds, {time_per_sample:.2f} sec/sample.")
#             batch = []
# 
#     # Process remaining samples in the last batch
#     if batch:
#         start_time = time.time()
#         sample_ids, metadata_texts = zip(*batch)
#         metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
#         failed_samples.extend(batch_failed_samples)
#         end_time = time.time()
# 
#         save_embeddings(metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
#         time_per_sample = (end_time - start_time) / len(batch)
#         total_samples_processed += len(batch)  # Increment the count
#         print(f"Processed last batch of {len(batch)} samples in {end_time - start_time:.2f} seconds, {time_per_sample:.2f} sec/sample.")
# 
# print(f"All {total_samples_processed} samples processed.")  # Print the total number of samples processed
# print("Failed samples:", failed_samples)
# =============================================================================



