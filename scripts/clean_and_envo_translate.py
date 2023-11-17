#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:08:44 2023

@author: dgaio
"""


# Description:  
# goes through raw metadata files, cleans it, and 
# replaces ontology labels at each occurrence (regex flexible)
# with their respectective description. 
# saves the metadata files with the same names followed by _clean before.txt 

import re
import os
import pandas as pd
import pickle
import datetime
import argparse
import concurrent.futures
import time
import glob 
import multiprocessing


# # runs for all samples (n=3769393) in ... minutes 
# python /Users/dgaio/github/metadata_mining/scripts/clean_and_envo_translate.py \
#     --path_to_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --metadata_dirs "sample.info_split_dirs"
    
# on atlas: 
# python github/metadata_mining/scripts/clean_and_envo_translate.py \
#     --path_to_dir "MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --metadata_dirs "sample.info_split_dirs"

def create_regex_pattern(label):
    match = re.match(r'([a-zA-Z]+)_?(\d+)', label)
    if not match:
        raise ValueError(f"Unexpected label format: {label}")

    prefix, digits = match.groups()
    pattern = re.compile(prefix + r'\D+' + digits, flags=re.IGNORECASE)
    return pattern


    
def process_samples(file_paths, compiled_patterns, label_info_dict, base_dir, compiled_endings):
    log_messages = []
    for file_path in file_paths:
        # Extract sample name from file path for logging
        sample = os.path.basename(file_path).replace('.txt', '')
        log_messages.append(f"\nProcessing file {sample}...")
        rejected_lines = []

        with open(file_path, 'r') as f:
            metadata = f.read()
            for label, joint_info in label_info_dict.items():
                pattern = compiled_patterns[label]  # Use the precompiled pattern
                matches = pattern.findall(metadata)
                if matches:
                    # Log each successful ontology conversion
                    for match in matches:
                        log_messages.append(f"Converting '{label}' to '{joint_info}' in file {sample}")
                    metadata = pattern.sub(joint_info, metadata)

            cleaned_metadata_lines = []
            for line in metadata.splitlines():
                stripped_line = line.strip()
                if stripped_line.lower().startswith(("experiment", "run")):
                    continue
                if any(compiled_end.search(stripped_line) for compiled_end in compiled_endings):
                    rejected_lines.append(stripped_line)
                    continue
                cleaned_metadata_lines.append(stripped_line)
            cleaned_metadata = "\n".join(cleaned_metadata_lines)

        clean_file_name = file_path.replace('.txt', '_clean.txt')
        with open(clean_file_name, 'w') as f:
            f.write(cleaned_metadata)
            f.write('\n')  # Ensure the file ends with a newline

        # Log rejected lines
        for line in rejected_lines:
            log_messages.append(f"rejected line: {line}")

    return log_messages, len(file_paths)

    

# =============================================================================
# def process_metadata(base_dir, label_info_dict, batch_size=10):
#     endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
#     compiled_endings = [re.compile(end, re.IGNORECASE) for end in endings_to_remove]
#     compiled_patterns = {label: create_regex_pattern(label) for label in label_info_dict.keys()}
# 
#     # Gather all .txt files except *_clean.txt
#     all_txt_files = glob.glob(os.path.join(base_dir, 'dirz_*', '*.txt'))
#     txt_files_to_process = [file for file in all_txt_files if not file.endswith('_clean.txt')]
# 
#     print(f"Total number of files to process: {len(txt_files_to_process)}")
#     sample_batches = [txt_files_to_process[i:i + batch_size] for i in range(0, len(txt_files_to_process), batch_size)]
#     print(f"Total number of batches: {len(sample_batches)}")
# 
#     now = datetime.datetime.now()
#     log_file_path = os.path.join(base_dir, f"log_clean_and_envo_translate_{now.strftime('%Y%m%d_%H%M%S')}.txt")
#     processed_samples_count = 0
# 
#     with concurrent.futures.ProcessPoolExecutor() as executor, open(log_file_path, 'a') as log_file:
#         futures = [executor.submit(process_samples, batch, compiled_patterns, label_info_dict, base_dir, compiled_endings) for batch in sample_batches]
#         for future in concurrent.futures.as_completed(futures):
#             log_messages, batch_sample_count = future.result()
#             for message in log_messages:
#                 log_file.write(message + "\n")
#             processed_samples_count += batch_sample_count
#             print(f"Processed {processed_samples_count} files")
# 
#     return processed_samples_count
# =============================================================================

# =============================================================================
# if __name__ == "__main__":
#     start_time = time.time()
# 
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_to_dir', required=True, help="Path to main directory.")
#     parser.add_argument('--ontology_dict', required=True, help="Pickle file with ontology dictionary.")
#     parser.add_argument('--metadata_dirs', required=True, help="Subdirectory for metadata files.")
#     args = parser.parse_args()
# 
#     base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
#     dict_file_path = os.path.join(args.path_to_dir, args.ontology_dict)
# 
#     with open(dict_file_path, 'rb') as f:
#         label_info_dict = pickle.load(f)
# 
#     processed_samples_count = process_metadata(base_dir, label_info_dict)
# 
#     print(f"Total processed files: {processed_samples_count}")
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Total execution time: {total_time:.2f} seconds")
# =============================================================================


def process_metadata(base_dir, label_info_dict, batch_size=10, max_workers=None):
    endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
    compiled_endings = [re.compile(end, re.IGNORECASE) for end in endings_to_remove]
    compiled_patterns = {label: create_regex_pattern(label) for label in label_info_dict.keys()}

    # Gather all .txt files except *_clean.txt
    all_txt_files = glob.glob(os.path.join(base_dir, 'dirz_*', '*.txt'))
    txt_files_to_process = [file for file in all_txt_files if not file.endswith('_clean.txt')]

    print(f"Total number of files to process: {len(txt_files_to_process)}")
    sample_batches = [txt_files_to_process[i:i + batch_size] for i in range(0, len(txt_files_to_process), batch_size)]
    print(f"Total number of batches: {len(sample_batches)}")

    now = datetime.datetime.now()
    log_file_path = os.path.join(base_dir, f"log_clean_and_envo_translate_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    processed_samples_count = 0

    # Set the number of workers if not specified
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() // 2  # Use half of the available cores

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor, open(log_file_path, 'a') as log_file:
        futures = [executor.submit(process_samples, batch, compiled_patterns, label_info_dict, base_dir, compiled_endings) for batch in sample_batches]
        for future in concurrent.futures.as_completed(futures):
            log_messages, batch_sample_count = future.result()
            for message in log_messages:
                log_file.write(message + "\n")
            processed_samples_count += batch_sample_count
            print(f"Processed {processed_samples_count} files")

    return processed_samples_count



if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dir', required=True, help="Path to main directory.")
    parser.add_argument('--ontology_dict', required=True, help="Pickle file with ontology dictionary.")
    parser.add_argument('--metadata_dirs', required=True, help="Subdirectory for metadata files.")
    parser.add_argument('--max_workers', type=int, default=None, help="Maximum number of worker processes to use.")
    args = parser.parse_args()

    base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
    dict_file_path = os.path.join(args.path_to_dir, args.ontology_dict)

    with open(dict_file_path, 'rb') as f:
        label_info_dict = pickle.load(f)

    processed_samples_count = process_metadata(base_dir, label_info_dict, max_workers=args.max_workers)

    print(f"Total processed files: {processed_samples_count}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")






