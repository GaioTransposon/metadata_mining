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
import gc 
import time


# # runs for all samples (n=3769393) in ... minutes 
# python /Users/dgaio/github/metadata_mining/scripts/clean_and_envo_translate.py \
#     --path_to_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --samples_dict "gold_dict.pkl" \
#     --metadata_dirs "sample.info_split_dirs"

def create_regex_pattern(label):
    match = re.match(r'([a-zA-Z]+)_?(\d+)', label)
    if not match:
        raise ValueError(f"Unexpected label format: {label}")

    prefix, digits = match.groups()
    pattern = re.compile(prefix + r'\D+' + digits, flags=re.IGNORECASE)
    return pattern


def process_samples(batch, compiled_patterns, label_info_dict, base_dir, compiled_endings):
    log_messages = []
    for sample in batch:
        log_messages.append(f"\nProcessing sample {sample}...")
        rejected_lines = []

        sub_dir = f"dir_{sample[-3:]}"
        sample_file = os.path.join(base_dir, sub_dir, sample + '.txt')
        with open(sample_file, 'r') as f:
            metadata = f.read()
            for label, joint_info in label_info_dict.items():
                pattern = compiled_patterns[label]  # Use the precompiled pattern
                matches = pattern.findall(metadata)
                if matches:
                    # Log each successful ontology conversion
                    for match in matches:
                        log_messages.append(f"Converting '{label}' to '{joint_info}' in sample {sample}")
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

        clean_file_name = sample_file.replace('.txt', '_clean.txt')
        with open(clean_file_name, 'w') as f:
            f.write(cleaned_metadata)
            f.write('\n')  # Ensure the file ends with a newline

        # Log rejected lines
        for line in rejected_lines:
            log_messages.append(f"rejected line: {line}")

    return log_messages, len(batch)



# =============================================================================
# def process_metadata(samples, label_info_dict, base_dir, batch_size=10):
#     endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
#     compiled_endings = [re.compile(end, re.IGNORECASE) for end in endings_to_remove]
# 
#     # Precompile regex patterns
#     compiled_patterns = {label: create_regex_pattern(label) for label in label_info_dict.keys()}
# 
#     now = datetime.datetime.now()
#     log_file_path = os.path.join(base_dir, f"log_clean_and_envo_translate_{now.strftime('%Y%m%d_%H%M%S')}.txt")
#     log_buffer = []
#     
#     # Creating batches of samples
#     sample_batches = [samples['sample'][i:i + batch_size] for i in range(0, len(samples['sample']), batch_size)]
#     print(f"Total number of batches: {len(sample_batches)}")
# 
#     processed_samples_count = 0  # Initialize the count of processed samples
# 
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(process_samples, batch, compiled_patterns, label_info_dict, base_dir, compiled_endings) for batch in sample_batches]
#         for future in concurrent.futures.as_completed(futures):
#             log_messages, batch_sample_count = future.result()  # Get log messages and count of samples in the batch
# 
#             for message in log_messages:
#                 log_buffer.append(message + "\n")
# 
#             processed_samples_count += batch_sample_count  # Update the total count with the count from each batch
#             print(f"Processed {processed_samples_count} samples")
# 
#     with open(log_file_path, 'w') as log_file:
#         log_file.writelines(log_buffer)
# 
#     return processed_samples_count
# =============================================================================

def process_metadata(samples, label_info_dict, base_dir, batch_size=10):
    endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
    compiled_endings = [re.compile(end, re.IGNORECASE) for end in endings_to_remove]

    # Precompile regex patterns
    compiled_patterns = {label: create_regex_pattern(label) for label in label_info_dict.keys()}

    now = datetime.datetime.now()
    log_file_path = os.path.join(base_dir, f"log_clean_and_envo_translate_{now.strftime('%Y%m%d_%H%M%S')}.txt")

    # Creating batches of samples
    sample_batches = [samples['sample'][i:i + batch_size] for i in range(0, len(samples['sample']), batch_size)]
    print(f"Total number of batches: {len(sample_batches)}")

    processed_samples_count = 0  # Initialize the count of processed samples

    with concurrent.futures.ProcessPoolExecutor() as executor, open(log_file_path, 'a') as log_file:
        futures = [executor.submit(process_samples, batch, compiled_patterns, label_info_dict, base_dir, compiled_endings) for batch in sample_batches]
        for future in concurrent.futures.as_completed(futures):
            log_messages, batch_sample_count = future.result()  # Get log messages and count of samples in the batch

            for message in log_messages:
                log_file.write(message + "\n")

            processed_samples_count += batch_sample_count  # Update the total count with the count from each batch
            print(f"Processed {processed_samples_count} samples")

    return processed_samples_count


if __name__ == "__main__":
    
    start_time = time.time()  # Record start time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dir', required=True, help="Path to main directory.")
    parser.add_argument('--ontology_dict', required=True, help="Pickle file with ontology dictionary.")
    parser.add_argument('--samples_dict', required=True, help="Pickle file with sample IDs.")
    parser.add_argument('--metadata_dirs', required=True, help="Subdirectory for metadata files.")
    args = parser.parse_args()

    base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
    dict_file_path = os.path.join(args.path_to_dir, args.ontology_dict)
    samples_dict_path = os.path.join(args.path_to_dir, args.samples_dict)

    with open(dict_file_path, 'rb') as f:
        label_info_dict = pickle.load(f)

    with open(samples_dict_path, 'rb') as f:
        gold_data = pickle.load(f)
        if isinstance(gold_data, dict):
            samples_dict = gold_data
        elif isinstance(gold_data, tuple):
            for item in gold_data:
                if isinstance(item, dict):
                    samples_dict = item
                    break

    # Add this line to print the number of samples in the dictionary
    print(f"Number of samples in samples_dict: {len(samples_dict)}")
    
    
    samples_df = pd.DataFrame({'sample': list(samples_dict.keys())})
    processed_samples_count = process_metadata(samples_df, label_info_dict, base_dir)


    print(f"Total processed samples: {processed_samples_count}")
    
    end_time = time.time()  # Record end time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")




