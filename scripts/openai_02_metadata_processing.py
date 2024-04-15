#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:54:13 2024

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
# PHASE 2: Metadata Processing 
# =======================================================

class MetadataProcessor:
    
    
    def __init__(self, work_dir, chunking, chunk_size, system_prompt_file, encoding_name):
        self.work_dir = work_dir
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.system_prompt_file = system_prompt_file
        self.encoding_name = encoding_name
        self.processed_sample_ids = []
        
    def load_metadata(self):
        """
        Loads the metadata dictionary from a pickle file named 'metadata_prov.pkl'
        located in the work directory.
        """
        metadata_path = os.path.join(self.work_dir, 'metadata_prov.pkl')
        try:
            with open(metadata_path, 'rb') as file:
                metadata_dict = pickle.load(file)
            return metadata_dict
        except FileNotFoundError:
            logging.error(f"Metadata file '{metadata_path}' not found.")
            return None
        except IOError:
            logging.error(f"Error reading metadata file '{metadata_path}'.")
            return None



    def process_metadata(self, missing_samples=None):
        metadata_dict = self.load_metadata()  
        if metadata_dict is None:
            logging.error("Failed to load metadata for processing.")
            return {}
        
        # If missing_samples is provided, filter the metadata_dict to include only those samples
        if missing_samples is not None:
            metadata_dict = {sample_id: metadata for sample_id, metadata in metadata_dict.items() if sample_id in missing_samples}
        
        processed_metadata = {}
        
        for sample_id, metadata in metadata_dict.items():
            self.processed_sample_ids.append(sample_id)
            cleaned_metadata_lines = [line.strip() for line in metadata.splitlines()]
            if cleaned_metadata_lines:
                processed_metadata[sample_id] = "\n".join(cleaned_metadata_lines)
                
        logging.info(f"Processed samples: {self.processed_sample_ids}")
        
        return processed_metadata


    def token_count(self, text):
        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    

    

    def load_system_prompt(self):
        prompt_file = os.path.join(self.work_dir, self.system_prompt_file)
        try:
            with open(prompt_file, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            logging.error(f"System prompt file '{prompt_file}' not found.")
            return None
        except IOError:
            logging.error(f"Error reading system prompt file '{prompt_file}'.")
            return None
        

    def save_chunks_to_file(self, chunks):
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d%H%M')
        filename = os.path.join(self.work_dir, f"metadata_chunks_{formatted_time}.txt")
        with open(filename, 'w') as file:
            for chunk in chunks:
                file.write(chunk)
                file.write("\n\n-----\n\n")
        logging.info(f"Saved metadata chunks to: {filename}")



    def first_fit_decreasing_bin(self, samples_with_tokens, effective_max_tokens):
        bins = []
        for sample_id, token_count in sorted(samples_with_tokens, key=lambda x: x[1], reverse=True):
            placed = False
            for bin in bins:
                if sum(token_count for _, token_count in bin) + token_count <= effective_max_tokens:
                    bin.append((sample_id, token_count))
                    placed = True
                    break
            if not placed and token_count <= effective_max_tokens:
                bins.append([(sample_id, token_count)])
        #print(bins)
        return bins


    

    def create_and_save_chunks(self, metadata_dict, return_ids=False):
        if self.chunking == "no":
            # When chunking is disabled, process each metadata entry individually
            chunks = []
            for sample_id, metadata in metadata_dict.items():
                metadata_token_count = self.token_count(f"'sample_ID={sample_id}': '{metadata}'")
                if metadata_token_count <= self.chunk_size:
                    chunks.append(f"'sample_ID={sample_id}': '{metadata}'")
                    self.processed_sample_ids.append(sample_id)
                else:
                    logging.warning(f"Sample ID {sample_id} with token count {metadata_token_count} exceeds the chunk size of {self.chunk_size} and will be excluded.")
        else:
            #print(f"My chunk size is: {self.chunk_size}")
            
            system_prompt_size = self.token_count(self.load_system_prompt())
            #print('System prompt size:', system_prompt_size)
            
            effective_max_tokens = self.chunk_size - system_prompt_size
            samples_with_tokens = [(sample_id, self.token_count(f"'sample_ID={sample_id}': '{metadata}'")) for sample_id, metadata in metadata_dict.items()]
    
            oversized_sample_ids = [sample_id for sample_id, token_count in samples_with_tokens if token_count > effective_max_tokens]
            for oversized_sample_id in oversized_sample_ids:
                print(f"{oversized_sample_id} is too large to fit into a chunk of effective chunk size {effective_max_tokens}")
                logging.info(f"'Sample_ID={oversized_sample_id}' exceeds the effective max tokens of {effective_max_tokens} and will be excluded.")
  
            self.processed_sample_ids = [sample_id for sample_id, _ in samples_with_tokens if sample_id not in oversized_sample_ids]
            print(f'Processed sample IDs in create_and_save_chunks() step: {len(self.processed_sample_ids)}')
   
            #binned_samples = self.first_fit_decreasing_bin(samples_with_tokens, effective_max_tokens)
            binned_samples = self.first_fit_decreasing_bin([(sample_id, token_count) for sample_id, token_count in samples_with_tokens if sample_id not in oversized_sample_ids], effective_max_tokens)
    
            # Log token sizes of bins
            total_sum_of_all_bins = sum(sum(token_count for _, token_count in bin) for bin in binned_samples)
            total_tokens = total_sum_of_all_bins + (system_prompt_size * len(binned_samples))
            logging.info(f"Total input tokens (including system prompt(s)): {total_tokens}")
    
            chunks = []
            for bin in binned_samples:
                chunk = '\n~~~\n'.join(f"'sample_ID={sample_id}': '{metadata_dict[sample_id]}'" for sample_id, _ in bin)
                chunks.append(chunk)
    

    
        self.save_chunks_to_file(chunks)
        if return_ids:
            return chunks, self.processed_sample_ids
        return chunks

    
    
    
        

        


# =============================================================================
# test = MetadataProcessor("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/", 
#                          "yes",
#                           2000, 
#                           "openai_system_prompt.txt", 
#                           "cl100k_base")
# 
# # option 2. test situation in which missing_samples are present
# missing_samples=["SRS6212978"] 
# processed_metadata = test.process_metadata(missing_samples)
# print(processed_metadata)
# 
# # option 1. test normal situation (without missing_samples)                      
# processed_metadata = test.process_metadata()
# print(processed_metadata)
# 
# 
# 
# chunks = test.create_and_save_chunks(processed_metadata)
# for i in chunks: 
#     print('#####')
#     print(i)
#     print('#####')
#     
#     
# test.save_chunks_to_file(chunks) 
# =============================================================================




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