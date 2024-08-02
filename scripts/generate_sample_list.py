#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:05:20 2024

@author: dgaio
"""

# generate_sample_list.py

# =============================================================================
# import os
# import random
# import argparse
# import logging
# 
# def setup_logging():
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# 
# def reservoir_sampling(directory, seed, max_samples):
#     random.seed(seed)
#     sample_ids = []
#     n = 0  # Total number of encountered sample files
# 
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             if filename.endswith("_clean.txt"):
#                 n += 1
#                 if len(sample_ids) < max_samples:
#                     sample_ids.append(filename.replace('_clean.txt', ''))
#                 else:
#                     s = random.randint(0, n-1)
#                     if s < max_samples:
#                         sample_ids[s] = filename.replace('_clean.txt', '')
#         logging.debug(f"Processed directory: {dirpath}, current sample size: {len(sample_ids)}")
#     
#     return sample_ids
# 
# def write_samples_to_file(sample_ids, output_file_path):
#     with open(output_file_path, 'w') as f:
#         for sample_id in sample_ids:
#             f.write(sample_id + '\n')
#     logging.info(f"Sample list saved to {output_file_path}")
# 
# def generate_sample_list(work_dir, directory_with_split_metadata, seed, output_file, max_samples):
#     setup_logging()
#     logging.info("Starting the sample list generation process...")
#     
#     directory = os.path.join(work_dir, directory_with_split_metadata)
#     logging.info(f"Looking for files in: {directory}")
# 
#     # Get random sample IDs using reservoir sampling
#     sample_ids = reservoir_sampling(directory, seed, max_samples)
# 
#     # Write sample IDs to the output file
#     output_file_path = os.path.join(work_dir, output_file)
#     write_samples_to_file(sample_ids, output_file_path)
# 
#     logging.info("Sample list generation completed successfully.")
# 
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Generate a list of random sample IDs for metadata preparation.")
#     parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
#     parser.add_argument('--directory_with_split_metadata', required=True, help='Directory with split metadata')
#     parser.add_argument('--seed', type=int, required=True, help='Random seed for shuffling samples')
#     parser.add_argument('--output_file', required=True, help='Output file path for the sample list')
#     parser.add_argument('--max_samples', type=int, required=True, help='Maximum number of samples to list')
#     return parser.parse_args()
# 
# if __name__ == "__main__":
#     args = parse_arguments()
#     generate_sample_list(args.work_dir, args.directory_with_split_metadata, args.seed, args.output_file, args.max_samples)
# 
# =============================================================================


# script: generate_sample_list.py


import os
import random
import argparse
import logging

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_sample_ids(directory):
    """ Fetches all sample IDs from the specified directory. """
    sample_ids = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith("_clean.txt"):
                sample_id = filename.replace('_clean.txt', '')
                sample_ids.append(sample_id)
        logging.debug(f"Processed directory: {dirpath}, current sample count: {len(sample_ids)}")
    return sample_ids

def shuffle_and_write_samples(sample_ids, seed, output_file_path):
    """ Shuffles the sample IDs using the given seed and writes them to a file. """
    random.seed(seed)
    random.shuffle(sample_ids)
    with open(output_file_path, 'w') as f:
        for sample_id in sample_ids:
            f.write(sample_id + '\n')
    logging.info(f"Shuffled sample list saved to {output_file_path}")

def generate_sample_list(work_dir, directory_with_split_metadata, seed, output_file):
    setup_logging()
    logging.info("Starting the sample list generation process...")

    directory = os.path.join(work_dir, directory_with_split_metadata)
    logging.info(f"Looking for files in: {directory}")

    # Collect all sample IDs
    sample_ids = fetch_all_sample_ids(directory)

    # Shuffle and write the sample IDs to the output file
    output_file_path = os.path.join(work_dir, output_file)
    shuffle_and_write_samples(sample_ids, seed, output_file_path)

    logging.info("Sample list generation completed successfully.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a randomized list of sample IDs for metadata preparation.")
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--directory_with_split_metadata', required=True, help='Directory with split metadata')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for shuffling samples')
    parser.add_argument('--output_file', required=True, help='Output file path for the sample list')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generate_sample_list(args.work_dir, args.directory_with_split_metadata, args.seed, args.output_file)









# python /Users/dgaio/github/metadata_mining/scripts/generate_sample_list.py \
#     --work_dir "MicrobeAtlasProject" \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --seed 22 \
#     --output_file "samples_list.txt" 