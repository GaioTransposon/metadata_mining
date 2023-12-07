#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:57:10 2023

@author: dgaio
"""


import os
import pandas as pd
import numpy as np
import tiktoken
import pickle
from datetime import datetime
import logging

# =======================================================
# PHASE 1: Metadata Processing
# =======================================================
   
class MetadataProcessor:
    
    def __init__(self, work_dir, input_gold_dict, n_samples_per_biome, chunk_size, seed, directory_with_split_metadata):
        self.work_dir = work_dir
        self.input_gold_dict = os.path.join(work_dir, input_gold_dict)
        self.n_samples_per_biome = n_samples_per_biome
        self.seed = seed
        self.chunk_size = chunk_size
        self.directory_with_split_metadata = os.path.join(work_dir, directory_with_split_metadata)
        

    def load_gold_dict(self):
        with open(self.input_gold_dict, 'rb') as file:
            input_gold_dict = pickle.load(file)
            return input_gold_dict[0]  # because the second item is the list of pmids

    def transform_gold_dict_to_df(self, input_gold_dict):
        gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
        gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
        gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])

        # Check if geo_coordinates and geo_text exist in the tuple and extract them
        gold_dict_df['geo_coordinates'] = gold_dict_df['tuple_data'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
        gold_dict_df['geo_text'] = gold_dict_df['tuple_data'].apply(lambda x: x[3] if len(x) > 3 else np.nan)

        gold_dict_df.drop(columns='tuple_data', inplace=True)
        return gold_dict_df

    def get_random_samples(self, gold_dict_df): 
        # Filter out 'unknown' biomes before sampling - at the moment we don't want to test/validate gpt for the classification of "unknown"
        filtered_df = gold_dict_df[gold_dict_df['curated_biome'] != 'unknown']
        random_samples = filtered_df.groupby('curated_biome').apply(lambda x: x.sample(n=self.n_samples_per_biome, random_state=self.seed)).reset_index(drop=True)
        return random_samples

    def fetch_metadata_from_sample(self, sample):
        folder_name = f"dir_{sample[-3:]}"
        folder_path = os.path.join(self.directory_with_split_metadata, folder_name)
        metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
        with open(metadata_file_path, 'r') as f:
            return f.read()

    def process_metadata(self, samples):
        shuffled_samples = samples.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        processed_samples_count = 0
        processed_samples_list = []  # To keep track of which samples have been processed
        
        metadata_dict = {}
        
        for _, row in shuffled_samples.iterrows():
            metadata = self.fetch_metadata_from_sample(row['sample'])
            
            print("Metadata for", row['sample'])
            
            processed_samples_count += 1
            processed_samples_list.append(row['sample'])
            print(f"Processed samples count: {processed_samples_count}")        
            
            cleaned_metadata_lines = []
            for line in metadata.splitlines():
                stripped_line = line.strip()  # strip whitespace
                cleaned_metadata_lines.append(stripped_line)
            cleaned_metadata = "\n".join(cleaned_metadata_lines)
            metadata_dict[row['sample']] = cleaned_metadata
            
            print("Cleaned metadata:")
            print(cleaned_metadata)
            print("===================================")
            
        logging.info(f"All processed samples: {processed_samples_list}")
        
        
        return metadata_dict

    def token_count(self, text, encoding_name="cl100k_base"):
        """Return the number of tokens in the text using tiktoken."""
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)

            
    def create_and_save_chunks(self, metadata_dict, encoding_name="cl100k_base"):
        max_tokens = self.chunk_size
        chunks = []
        current_chunk, current_token_count = [], 0
        current_chunk_samples = 0  # Track the number of samples in the current chunk
        max_samples_per_chunk = 0  # Track the maximum number of samples across all chunks
    
        logging.info(f"Max tokens allowed per chunk: {max_tokens}")

        too_large_samples = []  # Step 1: Initialize an empty list to store samples that are too large

        for sample_id, metadata in metadata_dict.items():
            item = f"'sample_ID={sample_id}': '{metadata}'"
            #item_tokens = self.token_count(item)
            item_tokens = self.token_count(item, encoding_name)
    
            print(f"Processing sample: {sample_id} with {item_tokens} tokens")  # Debugging output
    
            # Check if this item alone exceeds the max_tokens
            if item_tokens > max_tokens:
                logging.info(f"Item {sample_id} is too large to fit in a single chunk.")
                too_large_samples.append(sample_id)
                continue
    
            # If adding this item doesn't exceed the token limit, add it to current chunk
            if (current_token_count + item_tokens) <= max_tokens:
                current_chunk.append(item)
                current_token_count += item_tokens
                current_chunk_samples += 1  # Increment the sample count for the current chunk
            else:
                print(f"Chunk is full with {current_token_count} tokens. Saving and starting a new one.")  # Debugging output
                chunks.append('\n~~~\n'.join(current_chunk))  # Use ~~~ as a separator between samples in the same chunk
                max_samples_per_chunk = max(max_samples_per_chunk, current_chunk_samples)  # Update the maximum
                current_chunk, current_token_count = [item], item_tokens
                current_chunk_samples = 1  # Reset for the new chunk (this item is the first one)
    
        # Handle the last chunk
        if current_chunk:
            chunks.append('\n~~~\n'.join(current_chunk))
            max_samples_per_chunk = max(max_samples_per_chunk, current_chunk_samples)  # Update the maximum if needed
    
        logging.info(f"Number of chunks: {len(chunks)}")
        logging.info(f"The maximum number of items in a chunk is: {max_samples_per_chunk}")
        logging.info(f"Samples that exceeded chunk size: {too_large_samples}")
        
        # Get the current date and time
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d%H%M')
    
        # Create the filename
        filename = os.path.join(self.work_dir, f"metadata_chunks_{formatted_time}.txt")
    
        # Write the chunks to the file
        with open(filename, 'w') as f:
            for chunk in chunks:
                f.write(chunk)
                f.write("\n\n-----\n\n")  # Separator between chunks
    
        return chunks


    def run(self):
        gold_dict = self.load_gold_dict()
        gold_dict_df = self.transform_gold_dict_to_df(gold_dict)
        random_samples = self.get_random_samples(gold_dict_df)
        metadata_dict = self.process_metadata(random_samples)
        chunks = self.create_and_save_chunks(metadata_dict)
        return chunks
    









# =============================================================================
# # load an encoding by name.
# encoding = tiktoken.get_encoding("cl100k_base")
# 
# # automatically load the correct encoding for a given model name.
# #encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# 
# # Turn text into tokens 
# my_tokens = encoding.encode('''
# 'sample_ID=SRS5567190': '>SRS5567190
# sample_alias=trim.sRA46.fq
# sample_TAXON_ID=447426
# sample_SCIENTIFIC_NAME=human oral metagenome
# sample_host=Homo sapiens
# sample_isolate=human saliva37
# sample_host_age=50
# sample_biomaterial_provider=luoyubin
# sample_host_sex=female
# sample_isolation_source=saliva
# sample_BioSampleModel=Metagenome or environmental
# study=SRP226795
# study_STUDY_TITLE=human saliva metagenome Metagenome
# study_STUDY_ABSTRACT=sequencing of human saliva metagenome'
#                             ''')
# 
# # Turn tokens into text 
# my_tokens_to_text = encoding.decode(my_tokens)
# 
# 
# # Count tokens by counting the length of the list returned by .encode().
# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens
# 
# num_tokens_from_string(my_tokens_to_text, "cl100k_base")
# =============================================================================






