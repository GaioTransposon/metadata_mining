#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:40:14 2024

@author: dgaio
"""

# script: metadata_prep_for_production.py


import os
import argparse
import pickle
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch metadata for a range of samples and save to a .pkl file.")
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--sample_list_file', type=str, required=True, help='File containing list of samples')
    parser.add_argument('--directory_with_split_metadata', type=str, required=True, help='Directory with split metadata')
    parser.add_argument('--sample_range', type=str, required=True, help='Range of samples to process, formatted as start:end')
    parser.add_argument('--output_pkl', type=str, required=True, help='Output .pkl file path')
    return parser.parse_args()

def fetch_metadata(sample_ids, directory_with_split_metadata, work_dir):
    metadata_dict = {}
    for sample_id in sample_ids:
        folder_name = f"dir_{sample_id[-3:]}"  # Assumes directories are named with the last three characters of the sample ID
        folder_path = os.path.join(work_dir, directory_with_split_metadata, folder_name)
        metadata_file_path = os.path.join(folder_path, f"{sample_id}_clean.txt")
        try:
            with open(metadata_file_path, 'r') as file:
                metadata_dict[sample_id] = file.read()
        except Exception as e:
            logging.error(f"Failed to fetch metadata for sample {sample_id}: {e}")
    return metadata_dict

def save_metadata(metadata_dict, output_pkl_path):
    with open(output_pkl_path, 'wb') as file:
        pickle.dump(metadata_dict, file)
    logging.info(f"Metadata saved to {output_pkl_path}")

def main():
    args = parse_arguments()
    setup_logging()

    # Parse the sample range
    start, end = map(int, args.sample_range.split(':'))

    # Load the list of sample IDs
    sample_list_path = os.path.join(args.work_dir, args.sample_list_file)
    with open(sample_list_path, 'r') as file:
        all_samples = [line.strip() for line in file]

    # Extract the specified range (1-indexed, converting to 0-indexed)
    selected_samples = all_samples[start-1:end]

    # Fetch metadata for the selected samples
    metadata_dict = fetch_metadata(selected_samples, args.directory_with_split_metadata, args.work_dir)

    # Save the metadata to a .pkl file
    output_pkl_path = os.path.join(args.work_dir, args.output_pkl)
    save_metadata(metadata_dict, output_pkl_path)

if __name__ == "__main__":
    main()



# python /Users/dgaio/github/metadata_mining/scripts/metadata_prep_for_production.py \
#     --work_dir "MicrobeAtlasProject" \
#     --sample_list_file "samples_list.txt" \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --sample_range 1:3500 \
#     --output_pkl "metadataprov.pkl" 






# =============================================================================
# import pickle
# 
# # Load the metadata from the pickle file
# with open('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadataprov.pkl', 'rb') as file:
#     metadata = pickle.load(file)
# 
# # Print the metadata for inspection
# for sample_id, content in metadata.items():
#     print(f"Sample ID: {sample_id}")
#     print("Metadata:")
#     print(content)
#     print("-" * 40)  # Separator for readability
# =============================================================================
