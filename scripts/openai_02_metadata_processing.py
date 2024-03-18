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
import re


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
        self.processed_sample_ids = []


    def load_gold_dict(self):
        with open(self.input_gold_dict, 'rb') as file:
            input_gold_dict = pickle.load(file)
            return input_gold_dict[0]  # because the second item is the list of pmids


    def transform_gold_dict_to_df(self, input_gold_dict):
        gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
        gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
        gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])

        # check if geo_coordinates and geo_text exist in the tuple and extract them
        gold_dict_df['geo_coordinates'] = gold_dict_df['tuple_data'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
        gold_dict_df['geo_text'] = gold_dict_df['tuple_data'].apply(lambda x: x[3] if len(x) > 3 else np.nan)

        gold_dict_df.drop(columns='tuple_data', inplace=True)
        return gold_dict_df


    def get_random_samples(self, gold_dict_df): 
        # filter out 'unknown' biomes before sampling - at the moment we don't want to test/validate gpt for the classification of "unknown"
        #filtered_df = gold_dict_df[gold_dict_df['curated_biome'] != 'unknown']
        filtered_df = gold_dict_df
        
        random_samples = filtered_df.groupby('curated_biome').apply(lambda x: x.sample(n=self.n_samples_per_biome, random_state=self.seed)).reset_index(drop=True)
        return random_samples


    def fetch_metadata_from_sample(self, sample):
        folder_name = f"dir_{sample[-3:]}"
        folder_path = os.path.join(self.directory_with_split_metadata, folder_name)
        metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
        with open(metadata_file_path, 'r') as f:
            return f.read()
        
    
    def refetch_metadata_for_samples(self, sample_ids):
        refetched_metadata = {}
        for sample_id in sample_ids:
            try:
                metadata = self.fetch_metadata_from_sample(sample_id)
                refetched_metadata[sample_id] = metadata
            except Exception as e:
                logging.error(f"Failed to refetch metadata for sample {sample_id}: {e}")
    
        return refetched_metadata

    

    def filter_lines_for_coordinates(self, lines):
        coord_pattern = re.compile(r'(\d+(\.\d+)?\s*[NS]\s*\d+(\.\d+)?\s*[EW])|(\bcoord\w*\b)|(\blat(itude)?\b)|(\blon(gitude)?\b)', re.IGNORECASE)
    
        filtered_lines = []
        for line in lines:
            # First check for the coordinate pattern
            if coord_pattern.search(line):
                word_count = len(line.split())
                # Then check if the word count is <= 20
                if word_count <= 20:
                    filtered_lines.append(line)
        return filtered_lines


    def process_metadata(self, samples):
        
        # Check if 'coordinates' is in the system prompt
        system_prompt = self.load_system_prompt()
        filter_for_coordinates = 'coordinates' in system_prompt
        
        shuffled_samples = samples.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        processed_samples_count = 0
        processed_samples_list = []  
        
        metadata_dict = {}
        
        for _, row in shuffled_samples.iterrows():
            metadata = self.fetch_metadata_from_sample(row['sample'])
            
            #print("Metadata for", row['sample'])
            
            processed_samples_count += 1
            processed_samples_list.append(row['sample'])

            # to store processed sample ids
            self.processed_sample_ids.append(row['sample']) 

            
            if filter_for_coordinates:
                # Use the filter_lines_for_coordinates function to filter lines
                cleaned_metadata_lines = self.filter_lines_for_coordinates(metadata.splitlines())
            else:
                # If not filtering for coordinates, keep all lines
                cleaned_metadata_lines = [line.strip() for line in metadata.splitlines()]
                
            # Check if cleaned_metadata_lines is not empty before adding to metadata_dict
            if cleaned_metadata_lines:
                cleaned_metadata = "\n".join(cleaned_metadata_lines)
                metadata_dict[row['sample']] = cleaned_metadata
            else:
                continue
    
            print(f"Processed samples count: {processed_samples_count}")
            print("Cleaned metadata:")
            print(metadata_dict[row['sample']])
            print("===================================")
            
# =============================================================================
#             cleaned_metadata = "\n".join(cleaned_metadata_lines)
#             metadata_dict[row['sample']] = cleaned_metadata
#     
#             print("Cleaned metadata:")
#             print(cleaned_metadata)
#             print("===================================")
# =============================================================================
            
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
                file.write("\n\n-----\n\n")  # separator between chunks
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
                # if the token_count exceeds the effective_max_tokens, create a new bin 
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
        
        # print messages into log, if sample metadata exceeds effective_max_tokens
        for sample_id, token_count in samples_with_tokens:
            if token_count > effective_max_tokens:
                logging.info(f"'sample_ID={sample_id}' is too large to fit into a chunk of effective chunk size {effective_max_tokens}")
        
        binned_samples = self.first_fit_decreasing_bin(samples_with_tokens, effective_max_tokens)

        # print token sizes of bins
        #print("Bins with token sizes and their sums:")
        total_sum_of_all_bins = 0
        for bin in binned_samples:
            bin_token_sizes = [token_count for _, token_count in bin]
            sum_of_tokens = sum(bin_token_sizes)
            total_sum_of_all_bins += sum_of_tokens
            #print(f"Bin token sizes: {bin_token_sizes}, Sum: {sum_of_tokens}")

        # add the sums of all bin tokens and the system prompt tokens multiplied by the number of bins (this is the total input tokens)
        total_tokens = total_sum_of_all_bins + (system_prompt_size * len(binned_samples))
        logging.info(f"Total input tokens (including system prompt(s)): {total_tokens}")

        # create and save chunks
        consolidated_chunks = []
        for bin in binned_samples:
            chunk = '\n~~~\n'.join(f"'sample_ID={sample_id}': '{metadata_dict[sample_id]}'" for sample_id, _ in bin)
            consolidated_chunks.append(chunk)
        
        self.save_chunks_to_file(consolidated_chunks)
        return consolidated_chunks
    

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