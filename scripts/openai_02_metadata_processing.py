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
    
    
    def __init__(self, work_dir, input_gold_dict, n_samples_per_biome, chunk_size, system_prompt_file, encoding_name, seed, directory_with_split_metadata):
        self.work_dir = work_dir
        self.input_gold_dict = os.path.join(work_dir, input_gold_dict)
        self.n_samples_per_biome = n_samples_per_biome
        self.seed = seed
        self.system_prompt_file = system_prompt_file
        self.encoding_name = encoding_name
        self.directory_with_split_metadata = os.path.join(work_dir, directory_with_split_metadata)
        self.chunk_size = chunk_size  


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
            
            #print("Cleaned metadata:")
            #print(cleaned_metadata)
            print("===================================")
            
        logging.info(f"All processed samples: {processed_samples_list}")
        
        return metadata_dict


    def token_count(self, text, encoding_name):
        """Return the number of tokens in the text using tiktoken."""
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
          
    
    def load_system_prompt(self):
        """Load the system prompt from a text file."""
        if not isinstance(self.system_prompt_file, str):
            raise TypeError("system_prompt_file must be a string")
    
        prompt_file = os.path.join(self.work_dir, self.system_prompt_file)
        try:
            with open(prompt_file, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"System prompt file '{prompt_file}' not found.")
            return None
        except IOError:
            print(f"Error reading system prompt file '{prompt_file}'.")
            return None
        
        
    def save_chunks_to_file(self, chunks):
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d%H%M')
        filename = os.path.join(self.work_dir, f"metadata_chunks_{formatted_time}.txt")

        with open(filename, 'w') as file:
            for chunk in chunks:
                file.write(chunk)
                file.write("\n\n-----\n\n")  # Separator between chunks
        logging.info(f"Saved metadata chunks to: {filename}")
        

    def first_fit_decreasing_bin(self, samples_with_tokens, effective_max_tokens):
        samples_with_tokens.sort(key=lambda x: x[1], reverse=True)
        bins = []
        for sample_id, token_count in samples_with_tokens:
            placed = False
            for bin in bins:
                current_bin_size = sum(token_count for _, token_count in bin)
                if current_bin_size + token_count <= effective_max_tokens:
                    bin.append((sample_id, token_count))
                    placed = True
                    break
            if not placed:
                # If the token_count exceeds the effective_max_tokens, create a new bin for the sample.
                if token_count <= effective_max_tokens:
                    bins.append([(sample_id, token_count)])
        return bins
   

    def create_and_save_chunks(self, metadata_dict, encoding_name):
        

        system_prompt_size = self.token_count(self.load_system_prompt(), encoding_name)
        #print('system_prompt_size:', system_prompt_size)
        
        effective_max_tokens = self.chunk_size - system_prompt_size
        #print('self.chunk_size:', self.chunk_size)
        #print('effective_max_tokens:', effective_max_tokens)

        samples_with_tokens = [(sample_id, self.token_count(f"'sample_ID={sample_id}': '{metadata}'", encoding_name)) for sample_id, metadata in metadata_dict.items()]
        #print('samples_with_tokens:', samples_with_tokens)
        
        # Check for samples with token sizes exceeding effective_max_tokens
        for sample_id, token_count in samples_with_tokens:
            if token_count > effective_max_tokens:
                print(f"WARNING: 'sample_ID={sample_id}' is too large to fit into a chunk of effective chunk size {effective_max_tokens}")
                logging.warning(f"'sample_ID={sample_id}' is too large to fit into a chunk of effective chunk size {effective_max_tokens}")
        
        binned_samples = self.first_fit_decreasing_bin(samples_with_tokens, effective_max_tokens)

        # Print the bins with token sizes
        print("Bins with token sizes:")
        for bin in binned_samples:
            print([token_count for _, token_count in bin])

        # Create and save chunks
        chunks = []
        for bin in binned_samples:
            chunk = '\n~~~\n'.join(f"'sample_ID={sample_id}': '{metadata_dict[sample_id]}'" for sample_id, _ in bin)
            chunks.append(chunk)
        
        self.save_chunks_to_file(chunks)
        return chunks


    def run(self):
        gold_dict = self.load_gold_dict()
        gold_dict_df = self.transform_gold_dict_to_df(gold_dict)
        random_samples = self.get_random_samples(gold_dict_df)
        metadata_dict = self.process_metadata(random_samples)
        chunks = self.create_and_save_chunks(metadata_dict, self.encoding_name)
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


