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




# =============================================================================
# def create_regex_pattern(label):
#     match = re.match(r'([a-zA-Z]+)_?(\d+)', label)
#     if not match:
#         raise ValueError(f"Unexpected label format: {label}")
# 
#     prefix, digits = match.groups()
#     pattern = prefix + r'\D+' + digits
#     return pattern
# 
# 
# def process_metadata(samples, label_info_dict, base_dir):
#     shuffled_samples = samples.sample(frac=1).reset_index(drop=True)
#     processed_samples_count = 0
#     total_samples_processed = 0
#     rejected_lines_count = 0
#     successful_ontology_conversions = 0
#     processed_samples_list = []
#     endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
#     metadata_dict = {}
# 
#     now = datetime.datetime.now()
#     log_file_path = os.path.join(base_dir, f"log_clean_and_envo_translate_{now.strftime('%Y%m%d_%H%M%S')}.txt")
#     log_buffer = []
# 
#     def log_and_print(message):
#         print(message)
#         log_buffer.append(message + "\n")
# 
#     for _, row in shuffled_samples.iterrows():
#         total_samples_processed += 1
#         log_and_print(f"\nProcessing sample {total_samples_processed} ({row['sample']})...")
# 
#         sub_dir = f"dir_{row['sample'][-3:]}"
#         sample_file = os.path.join(base_dir, sub_dir, row['sample'] + '.txt')
#         with open(sample_file, 'r') as f:
#             metadata = f.read()
#             conversions_log = []
#             for label, joint_info in label_info_dict.items():
#                 pattern = create_regex_pattern(label)
#                 matches = re.findall(pattern, metadata, flags=re.IGNORECASE)
#                 if matches:
#                     successful_ontology_conversions += len(matches)
#                     conversions_log.append(f"'{label}' --> '{joint_info}'")
#                     metadata = re.sub(pattern, joint_info, metadata, flags=re.IGNORECASE)
# 
#             cleaned_metadata_lines = []
#             rejected_lines_log = []
#             for line in metadata.splitlines():
#                 stripped_line = line.strip()
#                 should_keep = True
#                 if stripped_line.lower().startswith(("experiment", "run", ">")):
#                     should_keep = False
#                 else:
#                     for ending in endings_to_remove:
#                         if re.search(ending, stripped_line, re.IGNORECASE):
#                             rejected_lines_count += 1
#                             rejected_lines_log.append(f"Rejected line (ends with {ending}): {stripped_line}")
#                             should_keep = False
#                             break
#                 if should_keep:
#                     cleaned_metadata_lines.append(stripped_line)
#             cleaned_metadata = "\n".join(cleaned_metadata_lines)
#             metadata_dict[row['sample']] = cleaned_metadata
# 
#             clean_file_name = sample_file.replace('.txt', '_clean.txt')
#             with open(clean_file_name, 'w') as f:
#                 f.write(cleaned_metadata)
# 
#             processed_samples_count += 1
#             processed_samples_list.append(row['sample'])
#             if conversions_log:
#                 log_and_print("Converting labels:")
#                 for conversion in conversions_log:
#                     log_and_print(conversion)
#             if rejected_lines_log:
#                 log_and_print("\nRejected lines:")
#                 for rejection in rejected_lines_log:
#                     log_and_print(rejection)
#             log_and_print("===================================")
# 
#     log_and_print(f"\nAll processed samples: {processed_samples_list}")
#     log_and_print(f"Processed samples count: {processed_samples_count}")
#     log_and_print(f"Total rejected lines: {rejected_lines_count}")
#     log_and_print(f"Successful ontology conversions: {successful_ontology_conversions}")
# 
#     with open(log_file_path, 'w') as log_file:
#         log_file.writelines(log_buffer)
#     return metadata_dict
# 
# 
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_to_dir', required=True, help="Path to main directory.")
#     parser.add_argument('--ontology_dict', required=True, help="Pickle file with ontology dictionary.")
#     parser.add_argument('--samples_dict', required=True, help="Pickle file with sample IDs.")
#     parser.add_argument('--metadata_dirs', required=True, help="Subdirectory for metadata files.")
#     args = parser.parse_args()
# 
#     # Construct paths
#     base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
#     dict_file_path = os.path.join(args.path_to_dir, args.ontology_dict)
#     samples_dict_path = os.path.join(args.path_to_dir, args.samples_dict)
# 
#     # Load ontology dictionary
#     with open(dict_file_path, 'rb') as f:
#         label_info_dict = pickle.load(f)
#     print('Ontology dictionary has ', len(label_info_dict), ' keys')
# 
#     
#     # Load samples dictionary (because my gold_dict is a tuple with the dict as first item of the tuple)
#     with open(samples_dict_path, 'rb') as f:
#         gold_data = pickle.load(f)
#     
#         # Check if the loaded data is a dictionary
#         if isinstance(gold_data, dict):
#             samples_dict = gold_data
#     
#         # If it's a tuple, iterate through its items and look for a dictionary
#         elif isinstance(gold_data, tuple):
#             for item in gold_data:
#                 if isinstance(item, dict):
#                     samples_dict = item
#                     break
#     
#     samples_df = pd.DataFrame({'sample': list(samples_dict.keys())})
#     
#     metadata_dict = process_metadata(samples_df, label_info_dict, base_dir)
# =============================================================================
    


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


# to do it in batches: (only 6 days)
# =============================================================================
# def process_samples(batch, compiled_patterns, label_info_dict, base_dir, compiled_endings):
#     log_messages = []
#     for sample in batch:
#         log_messages.append(f"\nProcessing sample ({sample})...")
# 
#         sub_dir = f"dir_{sample[-3:]}"
#         sample_file = os.path.join(base_dir, sub_dir, sample + '.txt')
#         with open(sample_file, 'r') as f:
#             metadata = f.read()
#             for label, joint_info in label_info_dict.items():
#                 pattern = compiled_patterns[label]  # Use the precompiled pattern
#                 if pattern.search(metadata):  # Check if the pattern is found before substitution
#                     metadata = pattern.sub(joint_info, metadata)
# 
#             cleaned_metadata_lines = []
#             for line in metadata.splitlines():
#                 stripped_line = line.strip()
#                 if stripped_line.lower().startswith(("experiment", "run")):
#                     continue
#                 if any(compiled_end.search(stripped_line) for compiled_end in compiled_endings):
#                     continue
#                 cleaned_metadata_lines.append(stripped_line)
#             cleaned_metadata = "\n".join(cleaned_metadata_lines)
# 
#         clean_file_name = sample_file.replace('.txt', '_clean.txt')
# 
#         #print(clean_file_name)
#         with open(clean_file_name, 'w') as f:
#             f.write(cleaned_metadata)
#             f.write('\n')  # Ensure the file ends with a newline
# 
#     del batch
#     gc.collect()  # Manually trigger garbage collection
#     return log_messages
# 
# 
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
#     processed_samples_count = 0
# 
#     # Creating batches of samples
#     sample_batches = [samples['sample'][i:i + batch_size] for i in range(0, len(samples['sample']), batch_size)]
#     print(f"Total number of batches: {len(sample_batches)}")
# 
#     
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(process_samples, batch, compiled_patterns, label_info_dict, base_dir, compiled_endings) for batch in sample_batches]
#         for future in concurrent.futures.as_completed(futures):
#             log_messages = future.result()
#             for message in log_messages:
#                 log_buffer.append(message + "\n")
#             processed_samples_count += len(log_messages)  # Corrected line
#             print(f"Processed {processed_samples_count} samples")
# 
#     with open(log_file_path, 'w') as log_file:
#         log_file.writelines(log_buffer)
# 
#     return processed_samples_count
# =============================================================================


def process_sample(sample, compiled_patterns, label_info_dict, base_dir, compiled_endings):
    log_messages = []
    log_messages.append(f"\nProcessing sample ({sample})...")

    sub_dir = f"dir_{sample[-3:]}"
    sample_file = os.path.join(base_dir, sub_dir, sample + '.txt')
    with open(sample_file, 'r') as f:
        metadata = f.read()
        for label, joint_info in label_info_dict.items():
            pattern = compiled_patterns[label]  # Use the precompiled pattern
            matches = pattern.findall(metadata)
            if matches:
                metadata = pattern.sub(joint_info, metadata)

        cleaned_metadata_lines = []
        for line in metadata.splitlines():
            stripped_line = line.strip()
            if stripped_line.lower().startswith(("experiment", "run")):
                continue
            if any(compiled_end.search(stripped_line) for compiled_end in compiled_endings):
                continue
            cleaned_metadata_lines.append(stripped_line)
        cleaned_metadata = "\n".join(cleaned_metadata_lines)

    clean_file_name = sample_file.replace('.txt', '_clean.txt')

    with open(clean_file_name, 'w') as f:
        f.write(cleaned_metadata)
        f.write('\n')  # Ensure the file ends with a newline

    return log_messages



def process_metadata(samples, label_info_dict, base_dir):
    endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
    compiled_endings = [re.compile(end, re.IGNORECASE) for end in endings_to_remove]
    processed_samples_count = 0

    # Precompile regex patterns
    compiled_patterns = {label: create_regex_pattern(label) for label in label_info_dict.keys()}

    now = datetime.datetime.now()
    log_file_path = os.path.join(base_dir, f"log_clean_and_envo_translate_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    log_buffer = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_sample, sample, compiled_patterns, label_info_dict, base_dir, compiled_endings) for sample in samples['sample']]
        for future in concurrent.futures.as_completed(futures):
            log_messages = future.result()
            for message in log_messages:
                log_buffer.append(message + "\n")
            processed_samples_count += 1
            print(f"Processed {processed_samples_count} samples")

    with open(log_file_path, 'w') as log_file:
        log_file.writelines(log_buffer)

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

    samples_df = pd.DataFrame({'sample': list(samples_dict.keys())})
    processed_samples_count = process_metadata(samples_df, label_info_dict, base_dir)


    print(f"Total processed samples: {processed_samples_count}")
    
    end_time = time.time()  # Record end time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


