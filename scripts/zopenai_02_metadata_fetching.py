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
# PHASE 1: Metadata Fetching
# =======================================================

class MetadataFetching:
    
    
    def __init__(self, work_dir, directory_with_split_metadata, input_gold_dict, n_samples_per_biome, seed):
        self.work_dir = work_dir
        self.directory_with_split_metadata = os.path.join(work_dir, directory_with_split_metadata)  
        self.input_gold_dict = os.path.join(work_dir, input_gold_dict)
        self.n_samples_per_biome = n_samples_per_biome
        self.seed = seed


    def load_gold_dict(self):
        with open(self.input_gold_dict, 'rb') as file:
            input_gold_dict = pickle.load(file)
            return input_gold_dict[0]  # these are the sample ids


    def transform_gold_dict_to_df(self, input_gold_dict):
        gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
        gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
        gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
        gold_dict_df['geo_coordinates'] = gold_dict_df['tuple_data'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
        gold_dict_df['geo_text'] = gold_dict_df['tuple_data'].apply(lambda x: x[3] if len(x) > 3 else np.nan)
        gold_dict_df.drop(columns='tuple_data', inplace=True)
        return gold_dict_df


    def get_random_samples(self, gold_dict_df):
        random_samples_df = gold_dict_df.groupby('curated_biome').apply(lambda x: x.sample(n=self.n_samples_per_biome, random_state=self.seed)).reset_index(drop=True)
        random_samples_list = random_samples_df.iloc[:, 0].tolist()  # Selecting the first column and converting to list
        return random_samples_list

    
    def fetch_metadata_for_samples(self, random_samples):
        """
        Fetches metadata for a list of sample IDs and returns a dictionary
        mapping sample IDs to their corresponding metadata.
        """
        metadata_dict = {}
        for sample_id in random_samples:
            
            print(sample_id)
            folder_name = f"dir_{sample_id[-3:]}"
            
            folder_path = os.path.join(self.directory_with_split_metadata, folder_name)
            metadata_file_path = os.path.join(folder_path, f"{sample_id}_clean.txt")
            try:
                with open(metadata_file_path, 'r') as file:
                    metadata_dict[sample_id] = file.read()
            except Exception as e:
                logging.error(f"Failed to fetch metadata for sample {sample_id}: {e}")
        return metadata_dict
    
    
    def save_metadata(self, metadata_dict):
        """
        Saves the metadata dictionary as a pickle file in the work directory.
        This will overwrite the file if it already exists.
        """
        output_path = os.path.join(self.work_dir, 'metadata_prov.pkl')
        with open(output_path, 'wb') as file:
            pickle.dump(metadata_dict, file)
        print(f"Metadata dictionary saved to {output_path}")




# =============================================================================
# test = MetadataFetching("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/", 
#                  "sample.info_split_dirs", 
#                  "gold_dict.pkl", 
#                  2, 
#                  32)
# # Load the gold dictionary
# gold_dict = test.load_gold_dict()
# # Transform the gold dictionary to a DataFrame
# gold_dict_df = test.transform_gold_dict_to_df(gold_dict)
# print(gold_dict_df)
# # Get random samples from the DataFrame
# random_samples = test.get_random_samples(gold_dict_df)
# print(random_samples)
# # Fetch metadata for the random samples
# metadata_dict = test.fetch_metadata_for_samples(random_samples)
# print(metadata_dict)
# # save 
# test.save_metadata(metadata_dict)
# =============================================================================








