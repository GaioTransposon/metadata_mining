#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:57:36 2024

@author: dgaio
"""

import argparse
import time
import os
import logging
from datetime import datetime
from openai_02_metadata_fetching import MetadataFetching
#from openai_02_metadata_processing import MetadataProcessor



def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare metadata for GPT interaction.')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--input_gold_dict', type=str, required=True, help='Input gold dictionary filename')
    parser.add_argument('--n_samples_per_biome', type=int, required=True, help='Number of samples per biome')
    parser.add_argument('--chunking', type=str, choices=['yes', 'no'], required=True, help='Enable or disable chunking of metadata')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of tokens per chunk')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for shuffling samples')
    parser.add_argument('--directory_with_split_metadata', type=str, required=True, help='Directory with split metadata')
    parser.add_argument('--system_prompt_file', type=str, required=True, help='it should be named openai_system_prompt.txt (location: github)')
    parser.add_argument('--encoding_name', type=str, required=True, help='Name of the tokenizer encoding')
    return parser.parse_args()


# Logging function: logs to both the console and a file
def setup_logging():
    
    # determine the dir of the running script - logging file will be save there
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_filename = datetime.now().strftime("metadata_preparation_%Y%m%d%H%M%S.log")
    log_filepath = os.path.join(script_directory, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # captures all logs at the DEBUG level and above

    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    ###
    # sets up logging to file and to console with different log levels, but a common formatter.

    # file handler for logging
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # captures INFO level and higher logs (i.e., WARNING, ERROR, CRITICAL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # captures WARNING level and higher logs (i.e., ERROR, CRITICAL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    ###
    



def main():
    args = parse_arguments()
    
    home_dir = os.getenv('HOME')
    work_dir = os.path.join(home_dir, args.work_dir)
    input_gold_dict = os.path.join(home_dir, args.input_gold_dict)
    #system_prompt_file = os.path.join(home_dir, args.system_prompt_file)
    
    
    setup_logging()
    
    # Metadata Fetching
    start_time = time.time()
    metadata_fetcher = MetadataFetching(work_dir, args.directory_with_split_metadata, input_gold_dict, args.n_samples_per_biome, args.chunking, args.chunk_size, args.seed)
    metadata_fetcher.run()
    end_time = time.time()
    logging.info(f"Metadata fetching time: {end_time - start_time} seconds")
    
# =============================================================================
#     # Metadata Processing
#     start_time = time.time()
#     metadata_processor = MetadataProcessor(work_dir, args.chunking, args.chunk_size, system_prompt_file, args.encoding_name)
#     processed_metadata = metadata_processor.process_metadata()
#     chunks, complete_sample_ids = metadata_processor.create_and_save_chunks(processed_metadata, return_ids=True)
#     metadata_processor.save_chunks_to_file(chunks)
#     end_time = time.time()
#     logging.info(f"Metadata processing time: {end_time - start_time} seconds")
# =============================================================================
    
if __name__ == "__main__":
    main()


# python /Users/dgaio/github/metadata_mining/scripts/metadata_preparation.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "github/metadata_mining/source_data/gold_dict.pkl" \
#     --n_samples_per_biome 1 \
#     --chunking "no" \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
#     --encoding_name "cl100k_base"
    

# python /Users/dgaio/github/metadata_mining/scripts/gpt_async_batch.py \
#     --work_dir "MicrobeAtlasProject" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 
    
