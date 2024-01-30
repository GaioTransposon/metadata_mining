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
import shutil
import pickle  


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Existing arguments
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path e.g.: "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" ')
    parser.add_argument('--metadata_directory', type=str, required=True, help='Directory with split metadata followed by "/" e.g.: "sample.info_split_dirs/" ') 
    parser.add_argument('--embeddings_file', type=str, required=True, help='name given to embeddings file e.g.: "embeddings.json" ')
    parser.add_argument('--temp_embeddings_file', type=str, required=True, help='name given to temporary embeddings file e.g.: "temp_embeddings.json" ')
    parser.add_argument('--batch_size', type=int, required=True, help='number of samples to process at a time (one API call) (nb: above 100 it doesn t go faster than 0.02 sec/sample)')
    parser.add_argument('--api_key_path', type=str, required=True, help='path to api key e.g.: "/Users/dgaio/my_api_key" ')
    
    # New optional arguments
    parser.add_argument('--get_for_gold_dict', type=str, default="no", help='Flag to process only samples in the gold dictionary. Options: "yes" or "no"')
    parser.add_argument('--gold_dict_path', type=str, default="gold_dict.pkl", help='Path to the gold dictionary file, relative to work_dir')

    return parser.parse_args()



def load_gold_dict(gold_dict_path):
    with open(gold_dict_path, 'rb') as file:
        gold_dict = pickle.load(file)
    return gold_dict[0]  # Assuming gold_dict is a tuple and the first element is the dictionary




# =============================================================================
# def get_embeddings(texts):
#     try:
#         response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
#         embeddings = [embedding['embedding'] for embedding in response['data']]
#         return np.array(embeddings), []
#     except Exception as e:
#         print(f"Error in batch embedding: {e}")
#         return np.array([]), texts
# =============================================================================
    

def get_embeddings(texts):
    try:
        start_api_call_time = time.time()  # Start timing for API call
        response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
        end_api_call_time = time.time()  # End timing for API call
        print(f"API call for {len(texts)} texts took {end_api_call_time - start_api_call_time:.2f} seconds")
        embeddings = [embedding['embedding'] for embedding in response['data']]
        return np.array(embeddings), []
    except Exception as e:
        print(f"Error in batch embedding: {e}")
        return np.array([]), texts
  
    
# creates empty json (if it doesn't exist)
def save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, embeddings, sample_ids):
    if not os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'w') as file:
            json.dump({}, file)  

    with open(EMBEDDINGS_FILE, 'r') as file:
        data = json.load(file)
        for sample_id, embedding in zip(sample_ids, embeddings):
            data[sample_id] = embedding.tolist()

    with open(TEMP_EMBEDDINGS_FILE, 'w') as temp_file:
        json.dump(data, temp_file)

    # replaces original file with updated temporary file
    shutil.move(TEMP_EMBEDDINGS_FILE, EMBEDDINGS_FILE)



def load_processed_samples(EMBEDDINGS_FILE):
    if not os.path.exists(EMBEDDINGS_FILE):
        return set()
    with open(EMBEDDINGS_FILE, 'r') as file:
        data = json.load(file)
    return set(data.keys())



# =============================================================================
# # ORI: 
# def main():
#     
#     args = parse_arguments()  
#     
#     work_dir = args.work_dir
#     METADATA_DIRECTORY = os.path.join(work_dir, args.metadata_directory)
#     EMBEDDINGS_FILE = os.path.join(work_dir, args.embeddings_file)
#     TEMP_EMBEDDINGS_FILE = os.path.join(work_dir, args.temp_embeddings_file)
#     BATCH_SIZE = args.batch_size
#     api_key_path = args.api_key_path
#     
#     
#     # load api key
#     with open(api_key_path, "r") as file:
#         openai.api_key = file.read().strip()
#         
#         
#     # main
#     processed_samples = load_processed_samples(EMBEDDINGS_FILE)
#     failed_samples = []
#     total_samples_processed_in_run = 0  
#     
#     for subdir in os.listdir(METADATA_DIRECTORY):
#         subdir_path = os.path.join(METADATA_DIRECTORY, subdir)
#         if not os.path.isdir(subdir_path):
#             continue
# 
#         sample_files = [f for f in os.listdir(subdir_path) if f.endswith('_clean.txt')]
#         batch = []
#         
#         for sample_file in sample_files:
#             sample_id = sample_file.split('_clean')[0]
#             if sample_id in processed_samples:
#                 continue
# 
#             metadata_file_path = os.path.join(subdir_path, sample_file)
#             with open(metadata_file_path, 'r') as file:
#                 metadata = file.read()
#             
#             batch.append((sample_id, metadata))
#             
#             if len(batch) >= BATCH_SIZE:
#                 start_time = time.time()
#                 sample_ids, metadata_texts = zip(*batch)
#                 metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
#                 failed_samples.extend(batch_failed_samples)
#                 end_time = time.time()
# 
#                 save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
#                 time_per_sample = (end_time - start_time) / len(batch)
#                 total_samples_processed_in_run += len(batch)  
#                 print(f"Processed last batch of {len(batch)} samples in {end_time - start_time:.2f} seconds, n={total_samples_processed_in_run} samples processed in run, {time_per_sample:.2f} sec/sample.")
#                 batch = []
# 
#         if batch:
#             start_time = time.time()
#             sample_ids, metadata_texts = zip(*batch)
#             metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
#             failed_samples.extend(batch_failed_samples)
#             end_time = time.time()
# 
#             save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
#             time_per_sample = (end_time - start_time) / len(batch)
#             print(f"Processed last batch of {len(batch)} samples in {end_time - start_time:.2f} seconds, {time_per_sample:.2f} sec/sample.")
# 
#     print("All samples processed.")
#     print("Failed samples:", failed_samples)
#     # instead of "failed_samples" list, we can add a snippet that checks which samples *_clean.txt are not in embeddings.json and runs them. 
# 
# =============================================================================


# =============================================================================
# # second:
# def main():
#     
#     args = parse_arguments()  
#     
#     work_dir = args.work_dir
#     METADATA_DIRECTORY = os.path.join(work_dir, args.metadata_directory)
#     EMBEDDINGS_FILE = os.path.join(work_dir, args.embeddings_file)
#     TEMP_EMBEDDINGS_FILE = os.path.join(work_dir, args.temp_embeddings_file)
#     BATCH_SIZE = args.batch_size
#     api_key_path = args.api_key_path
#     get_for_gold_dict = args.get_for_gold_dict.lower() == 'yes'
#     gold_dict_path = os.path.join(work_dir, args.gold_dict_path)
#     
#     # load api key
#     with open(api_key_path, "r") as file:
#         openai.api_key = file.read().strip()
# 
#     gold_dict = {}
#     if get_for_gold_dict:
#         gold_dict = load_gold_dict(gold_dict_path)
#     
#     # main
#     processed_samples = load_processed_samples(EMBEDDINGS_FILE)
#     failed_samples = []
#     total_samples_processed_in_run = 0  
#     
#     for subdir in os.listdir(METADATA_DIRECTORY):
#         subdir_path = os.path.join(METADATA_DIRECTORY, subdir)
#         if not os.path.isdir(subdir_path):
#             continue
# 
#         sample_files = [f for f in os.listdir(subdir_path) if f.endswith('_clean.txt')]
#         batch = []
#         
#         for sample_file in sample_files:
#             sample_id = sample_file.split('_clean')[0]
#             
#             # If get_for_gold_dict is True, skip samples not in gold_dict
#             if get_for_gold_dict and sample_id not in gold_dict:
#                 continue
#             
#             if sample_id in processed_samples:
#                 continue
# 
#             metadata_file_path = os.path.join(subdir_path, sample_file)
#             with open(metadata_file_path, 'r') as file:
#                 metadata = file.read()
#             
#             batch.append((sample_id, metadata))
#             
#             if len(batch) >= BATCH_SIZE:
#                 start_time = time.time()
#                 sample_ids, metadata_texts = zip(*batch)
#                 metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
#                 failed_samples.extend(batch_failed_samples)
#                 end_time = time.time()
# 
#                 save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
#                 time_per_sample = (end_time - start_time) / len(batch)
#                 total_samples_processed_in_run += len(batch)  
#                 print(f"Processed last batch of {len(batch)} samples in {end_time - start_time:.2f} seconds, n={total_samples_processed_in_run} samples processed in run, {time_per_sample:.2f} sec/sample.")
#                 batch = []
# 
#         if batch:
#             start_time = time.time()
#             sample_ids, metadata_texts = zip(*batch)
#             metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
#             failed_samples.extend(batch_failed_samples)
#             end_time = time.time()
# 
#             save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
#             time_per_sample = (end_time - start_time) / len(batch)
#             print(f"Processed last batch of {len(batch)} samples in {end_time - start_time:.2f} seconds, {time_per_sample:.2f} sec/sample.")
# 
#     print("All samples processed.")
#     print("Failed samples:", failed_samples)
#     # instead of "failed_samples" list, we can add a snippet that checks which samples *_clean.txt are not in embeddings.json and runs them. 
# 
# =============================================================================


def main():
    args = parse_arguments()  

    work_dir = args.work_dir
    METADATA_DIRECTORY = os.path.join(work_dir, args.metadata_directory)
    EMBEDDINGS_FILE = os.path.join(work_dir, args.embeddings_file)
    TEMP_EMBEDDINGS_FILE = os.path.join(work_dir, args.temp_embeddings_file)
    BATCH_SIZE = args.batch_size
    api_key_path = args.api_key_path
    get_for_gold_dict = args.get_for_gold_dict.lower() == 'yes'
    gold_dict_path = os.path.join(work_dir, args.gold_dict_path)
    
    # Load API key
    with open(api_key_path, "r") as file:
        openai.api_key = file.read().strip()

    gold_dict = {}
    if get_for_gold_dict:
        gold_dict = load_gold_dict(gold_dict_path)
    
    processed_samples = load_processed_samples(EMBEDDINGS_FILE)
    failed_samples = []
    total_samples_processed_in_run = 0  

    for subdir in os.listdir(METADATA_DIRECTORY):
        subdir_path = os.path.join(METADATA_DIRECTORY, subdir)
        if not os.path.isdir(subdir_path):
            continue

        sample_files = [f for f in os.listdir(subdir_path) if f.endswith('_clean.txt')]
        batch = []
        
        for sample_file in sample_files:
            sample_id = sample_file.split('_clean')[0]
            
            if get_for_gold_dict and sample_id not in gold_dict:
                continue
            
            if sample_id in processed_samples:
                continue

            metadata_file_path = os.path.join(subdir_path, sample_file)
            start_read_time = time.time()  # Start timing for reading file
            with open(metadata_file_path, 'r') as file:
                metadata = file.read()
            end_read_time = time.time()  # End timing for reading file
            print(f"Time taken to read {sample_file}: {end_read_time - start_read_time:.2f} seconds")

            batch.append((sample_id, metadata))
            
            if len(batch) >= BATCH_SIZE:
                start_batch_time = time.time()  # Start timing for batch processing
                sample_ids, metadata_texts = zip(*batch)
                metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
                failed_samples.extend(batch_failed_samples)
                save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
                end_batch_time = time.time()  # End timing for batch processing
                time_per_sample = (end_batch_time - start_batch_time) / len(batch)
                total_samples_processed_in_run += len(batch)
                print(f"Processed batch of {len(batch)} samples in {end_batch_time - start_batch_time:.2f} seconds, n={total_samples_processed_in_run} samples processed in run, {time_per_sample:.2f} sec/sample.")
                batch = []

        # Handle the last batch if it exists
        if batch:
            start_batch_time = time.time()  # Start timing for last batch processing
            sample_ids, metadata_texts = zip(*batch)
            metadata_embeddings, batch_failed_samples = get_embeddings(metadata_texts)
            failed_samples.extend(batch_failed_samples)
            save_embeddings(EMBEDDINGS_FILE, TEMP_EMBEDDINGS_FILE, metadata_embeddings, [id for id, _ in batch if id not in batch_failed_samples])
            end_batch_time = time.time()  # End timing for last batch processing
            time_per_sample = (end_batch_time - start_batch_time) / len(batch)
            print(f"Processed last batch of {len(batch)} samples in {end_batch_time - start_batch_time:.2f} seconds, {time_per_sample:.2f} sec/sample.")

    print("All samples processed.")
    print("Failed samples:", failed_samples)

    
    
if __name__ == "__main__":
    main()

        
        





# python /Users/dgaio/github/metadata_mining/scripts/embeddings_from_metadata.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --metadata_directory "sample.info_split_dirs/" \
#     --embeddings_file "embeddings_gd.json" \
#     --temp_embeddings_file "temp_embeddings_gd.json" \
#     --batch_size 10 \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --get_for_gold_dict "yes" \
#     --gold_dict_path "gold_dict.pkl"


# ssh dgaio@phobos.mls.uzh.ch
# python /mnt/mnemo5/dgaio/github/metadata_mining/scripts/embeddings_from_metadata.py \
#     --work_dir "/mnt/mnemo5/dgaio/MicrobeAtlasProject/" \
#     --metadata_directory "sample.info_split_dirs/" \
#     --embeddings_file "embeddings_gd.json" \
#     --temp_embeddings_file "temp_embeddings_gd.json" \
#     --batch_size 100 \
#     --api_key_path "/mnt/mnemo5/dgaio/my_api_key" \
#     --get_for_gold_dict "no" \
#     --gold_dict_path "gold_dict.pkl"
























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



