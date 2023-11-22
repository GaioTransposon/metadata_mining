#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:42:04 2023

@author: dgaio
"""

import os
import pickle
import argparse
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Process, cpu_count


##############################################################################
# # before running script, change ulimit in the session: 
# # $ ulimit -n 200000 <-- it's an estimation derived from: 
# # 40 (dirs and cpus at a time) * 3800 (files per dir) = 152000 --> round up: 200000
# # then run on local: 
# python /Users/dgaio/github/metadata_mining/scripts/clean_and_envo_translate2.py \
#     --path_to_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --metadata_dirs "sample.info_split_dirs" \
#     --max_processes 8
##
# # or on atlas: 
# python github/metadata_mining/scripts/clean_and_envo_translate2.py \
#     --path_to_dir "MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --metadata_dirs "sample.info_split_dirs" \
#     --max_processes 40
##############################################################################


def modify_ontology_dict(ontology_dict):
    modified_dict = {}
    for term, desc in ontology_dict.items():
        namespace, _, numeric_part = term.partition('_')
        namespace = namespace.lower()  # Convert namespace to lower case
        if numeric_part not in modified_dict:
            modified_dict[numeric_part] = {}
        modified_dict[numeric_part][namespace] = desc
    return modified_dict


def process_directory(dir_path, modified_ontology_dict, base_log_file_path):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.basename(dir_path)
    log_file_path = f"{base_log_file_path}_log_{dir_name}_{current_time}.txt"

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt") and not file_name.endswith("_clean.txt"):
            process_file(os.path.join(dir_path, file_name), modified_ontology_dict, log_file_path)


def process_file(file_path, modified_ontology_dict, log_file_path):
    try:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Processing file {os.path.basename(file_path)}...\n")

        new_lines = []
        with open(file_path, 'r') as file:
            for line in file:
                original_line = line  # Keep the original line for logging purposes
                line_lower = line.lower().strip()
                rejection_endings = ("=", "nan", "not applicable", "missing", " na", " na\n")

                # Check if the line should be rejected
                if line_lower.endswith(rejection_endings) or line_lower.startswith(("experiment", "run")):
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Rejected line: {line}")
                    continue  # Skip the rest of the processing for this line and do not add it to new_lines

                # Split line on first equals sign, if it exists
                parts = line.split('=', 1)
                prefix = parts[0] + '=' if len(parts) > 1 else ''
                line_to_process = parts[-1]

                for word in line_to_process.split():
                    numeric_part = ''.join(filter(str.isdigit, word))
                    word_lower = word.lower()  # Convert word to lower case for matching
                    if numeric_part in modified_ontology_dict:
                        for namespace, desc in modified_ontology_dict[numeric_part].items():
                            if namespace in word_lower:
                                replacement = f"'{desc}'"
                                line_to_process = line_to_process.replace(word, replacement)
                                with open(log_file_path, 'a') as log_file:
                                    log_file.write(f"Converting '{word}' in line '{original_line.strip()}' to '{replacement}'\n")
                                break  # Stop checking other namespaces for this word

                new_lines.append(prefix + line_to_process)

        with open(log_file_path, 'a') as log_file:
            log_file.write("\n")

        new_file_path = file_path.replace(".txt", "_clean.txt")
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(new_lines)

    except Exception as e:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Error processing file {file_path}: {e}\n")



def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dir", required=True)
    parser.add_argument("--ontology_dict", required=True)
    parser.add_argument("--metadata_dirs", required=True)
    parser.add_argument("--max_processes", type=int, default=5, help="Maximum number of concurrent processes - these are cpus hence directories, in this case")
    args = parser.parse_args()

    ontology_dict_path = os.path.join(args.path_to_dir, args.ontology_dict)
    with open(ontology_dict_path, 'rb') as f:
        ontology_dict = pickle.load(f)

    # Modify the ontology dictionary
    modified_ontology_dict = modify_ontology_dict(ontology_dict)

    base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
    base_log_file_path = os.path.join(base_dir, "log_clean_and_envo_translate")

    all_dir_names = [dir_name for dir_name in os.listdir(base_dir) if dir_name.startswith("dir_")]
    max_processes = min(args.max_processes, multiprocessing.cpu_count())

    for i in range(0, len(all_dir_names), max_processes):
        processes = []
        for dir_name in all_dir_names[i:i + max_processes]:
            dir_path = os.path.join(base_dir, dir_name)
            p = multiprocessing.Process(target=process_directory, args=(dir_path, modified_ontology_dict, base_log_file_path))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()  # Wait for the current batch of processes to complete before starting the next batch

    end_time = time.time()
    print(f"Script executed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()





