#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:42:04 2023

@author: dgaio
"""
# =============================================================================
# import os
# import pickle
# import argparse
# import multiprocessing
# import time
# from datetime import datetime
# import re
# 
# def process_directory(dir_path, ontology_dict, base_log_file_path):
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     dir_name = os.path.basename(dir_path)
#     log_file_path = f"{base_log_file_path}_log_{dir_name}_{current_time}.txt"
# 
#     for file_name in os.listdir(dir_path):
#         if file_name.endswith(".txt") and not file_name.endswith("_clean.txt"):
#             process_file(os.path.join(dir_path, file_name), ontology_dict, log_file_path)
# 
# def process_file(file_path, ontology_dict, log_file_path):
#     try:
#         with open(log_file_path, 'a') as log_file:
#             log_file.write(f"Processing file {os.path.basename(file_path)}...\n")
# 
#         new_lines = []
#         with open(file_path, 'r') as file:
#             for line in file:
#                 if line.startswith("experiment") or line.startswith("study"):
#                     with open(log_file_path, 'a') as log_file:
#                         log_file.write(f"rejected line: {line}")
#                 else:
#                     for term, desc in ontology_dict.items():
#                         if term in line:
#                             line = line.replace(term, f"'{desc}'")
#                             with open(log_file_path, 'a') as log_file:
#                                 log_file.write(f"Converting '{term}' to '{desc}' in file {os.path.basename(file_path)}\n")
#                     new_lines.append(line)
# 
#         with open(log_file_path, 'a') as log_file:
#             log_file.write("\n")
# 
#         new_file_path = file_path.replace(".txt", "_clean.txt")
#         with open(new_file_path, 'w') as new_file:
#             new_file.writelines(new_lines)
# 
#     except Exception as e:
#         with open(log_file_path, 'a') as log_file:
#             log_file.write(f"Error processing file {file_path}: {e}\n")
# 
# 
# 
# def main():
#     start_time = time.time()
# 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path_to_dir", required=True)
#     parser.add_argument("--ontology_dict", required=True)
#     parser.add_argument("--metadata_dirs", required=True)
#     args = parser.parse_args()
# 
#     ontology_dict_path = os.path.join(args.path_to_dir, args.ontology_dict)
#     with open(ontology_dict_path, 'rb') as f:
#         ontology_dict = pickle.load(f)
# 
#     base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
#     base_log_file_path = os.path.join(base_dir, "log_clean_and_envo_translate")
# 
#     processes = []
#     for dir_name in os.listdir(base_dir):
#         if dir_name.startswith("dirzz_"):
#             dir_path = os.path.join(base_dir, dir_name)
#             p = multiprocessing.Process(target=process_directory, args=(dir_path, ontology_dict, base_log_file_path))
#             processes.append(p)
#             p.start()
# 
#     for p in processes:
#         p.join()
# 
#     end_time = time.time()
#     print(f"Script executed in {end_time - start_time:.2f} seconds")
# 
# if __name__ == "__main__":
#     main()
# =============================================================================

import os
import pickle
import argparse
import multiprocessing
import time
from datetime import datetime



def modify_ontology_dict(ontology_dict):
    modified_dict = {}
    for term, desc in ontology_dict.items():
        modified_term = term.replace("ENVO_", "")  # Remove 'ENVO_' prefix
        modified_dict[modified_term] = desc
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
                line_lower = line.lower().strip()
                rejection_endings = ("=", "nan", "not applicable", "missing", " na", " na\n")

                # Check if the line should be rejected
                if line_lower.endswith(rejection_endings) or line_lower.startswith(("experiment", "run")):
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"rejected line: {line}")
                    continue  # Skip the rest of the processing for this line

                for term, desc in modified_ontology_dict.items():
                    if term in line:
                        line = line.replace(term, f"'{desc}'")
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"Converting '{term}' to '{desc}'\n")
                new_lines.append(line)

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
    args = parser.parse_args()

    ontology_dict_path = os.path.join(args.path_to_dir, args.ontology_dict)
    with open(ontology_dict_path, 'rb') as f:
        ontology_dict = pickle.load(f)

    # Modify the ontology dictionary to remove the 'ENVO_' prefix
    modified_ontology_dict = modify_ontology_dict(ontology_dict)

    base_dir = os.path.join(args.path_to_dir, args.metadata_dirs)
    base_log_file_path = os.path.join(base_dir, "log_clean_and_envo_translate")

    processes = []
    for dir_name in os.listdir(base_dir):
        if dir_name.startswith("dir_80"):
            dir_path = os.path.join(base_dir, dir_name)
            p = multiprocessing.Process(target=process_directory, args=(dir_path, modified_ontology_dict, base_log_file_path))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"Script executed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()












# python /Users/dgaio/github/metadata_mining/scripts/clean_and_envo_translate2.py \
#     --path_to_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --metadata_dirs "sample.info_split_dirs"