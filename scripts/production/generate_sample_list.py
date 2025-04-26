#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:05:20 2024

@author: dgaio
"""

# generate_sample_list.py


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


def load_whitelist(work_dir, whitelist_file):
    """Loads a whitelist of sample IDs from a TSV or text file."""
    whitelist_file = os.path.join(work_dir, whitelist_file)
    with open(whitelist_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def shuffle_and_write_samples(sample_ids, seed, output_file_path):
    """Shuffles the sample IDs using the given seed and writes them to a file."""
    random.seed(seed)
    random.shuffle(sample_ids)
    with open(output_file_path, 'w') as f:
        for sample_id in sample_ids:
            f.write(sample_id + '\n')
    logging.info(f"Shuffled sample list saved to {output_file_path}")


def generate_sample_list(work_dir, directory_with_split_metadata, seed, output_file, whitelist_file=None):
    setup_logging()
    logging.info("Starting the sample list generation process...")

    directory = os.path.join(work_dir, directory_with_split_metadata)
    logging.info(f"Looking for files in: {directory}")

    # Collect all sample IDs
    sample_ids = fetch_all_sample_ids(directory)
    logging.info(f"Found {len(sample_ids)} total sample IDs.")

    # Shuffle the full list
    random.seed(seed)
    random.shuffle(sample_ids)
    
    work_dir = os.path.expanduser(os.path.join("~", work_dir))

    # Filter with whitelist if provided
    if whitelist_file:
        whitelist = load_whitelist(work_dir, whitelist_file)
        filtered_sample_ids = [sid for sid in sample_ids if sid in whitelist]
        logging.info(f"Filtered down to {len(filtered_sample_ids)} sample IDs after applying whitelist.")
    else:
        filtered_sample_ids = sample_ids

    # Write final list to file
    output_file_path = os.path.join(work_dir, output_file)
    with open(output_file_path, 'w') as f:
        for sample_id in filtered_sample_ids:
            f.write(sample_id + '\n')

    logging.info(f"Final sample list written to: {output_file_path}")
    logging.info("Sample list generation completed successfully.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a randomized list of sample IDs for metadata preparation.")
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--directory_with_split_metadata', required=True, help='Directory with split metadata')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for shuffling samples')
    parser.add_argument('--output_file', required=True, help='Output file path for the sample list')
    parser.add_argument('--whitelist_file', type=str, required=False, help='Optional file with sample IDs to keep')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generate_sample_list(
        args.work_dir,
        args.directory_with_split_metadata,
        args.seed,
        args.output_file,
        args.whitelist_file
    )










# python /Users/danielagaio/github/metadata_mining/scripts/production/generate_sample_list.py \
#     --work_dir "cloudstor/Gaio/MicrobeAtlasProject" \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --seed 22 \
#     --output_file "samples_list_202504.txt" \
#     --whitelist_file "all_minfilt_sampids_map2021.tsv"







