#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:42:04 2023

@author: dgaio
"""


import os
import re
import pickle
import argparse
import multiprocessing
import time
from datetime import datetime


##############################################################################
# before running script, change ulimit in the session:
# $ ulimit -n 200000 <-- it's an estimation derived from:
# 40 (dirs and cpus at a time) * 3800 (files per dir) = 152000 --> round up: 200000
#
# then run on local:
# python ~/github/metadata_mining/scripts/clean_and_envo_translate.py \
#     --path_to_dir "~/MicrobeAtlasProject" \
#     --ontology_dict "ontologies_dict.pkl" \
#     --metadata_dirs "sample_info_split_dirs" \
#     --max_processes 8
#
# or on atlas: --max_processes 40
##############################################################################

MIN_LENGTH_FOR_PATTERN_CHECK = 1000
TRUNCATION_LABEL = " [TRUNCATED_REPEATING_STRUCTURE]"


def modify_ontology_dict(ontology_dict):
    modified_dict = {}
    for term, desc in ontology_dict.items():
        namespace, _, numeric_part = term.partition('_')
        namespace = namespace.lower()  # Convert namespace to lower case
        if numeric_part not in modified_dict:
            modified_dict[numeric_part] = {}
        modified_dict[numeric_part][namespace] = desc
    return modified_dict


def token_shape(token):
    """
    Convert a token into a coarse structural class.
    """
    stripped = token.strip().strip(".,;:()[]{}'\"")

    if not stripped:
        return "PUNCT"

    if re.fullmatch(r"\d+", stripped):
        return "NUM"

    if re.fullmatch(r"\d+\.\d+", stripped):
        return "FLOAT"

    if re.fullmatch(r"[A-Za-z]+[A-Za-z0-9]*[_-][A-Za-z0-9_-]+", stripped):
        return "ID"

    if re.fullmatch(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+", stripped):
        return "ALNUM"

    if re.fullmatch(r"[A-Za-z]+", stripped):
        return "WORD"

    return "OTHER"


def find_repeating_shape_start(text, window_size=7, min_repeats=3):
    """
    Detect repeated token-shape windows and return the approximate character
    position where repeating structured content starts.
    Returns None if no convincing repeated pattern is found.
    """
    tokens = text.split()
    if len(tokens) < window_size * min_repeats:
        return None

    shapes = [token_shape(token) for token in tokens]
    pattern_positions = {}

    for i in range(len(shapes) - window_size + 1):
        pattern = tuple(shapes[i:i + window_size])

        # Ignore all-WORD patterns to reduce false positives from normal prose
        if all(shape == "WORD" for shape in pattern):
            continue

        pattern_positions.setdefault(pattern, []).append(i)

    repeated_patterns = {
        pattern: positions
        for pattern, positions in pattern_positions.items()
        if len(positions) >= min_repeats
    }

    if not repeated_patterns:
        return None

    best_pattern = None
    best_positions = None
    best_score = -1

    for pattern, positions in repeated_patterns.items():
        unique_labels = len(set(pattern))
        non_word_count = sum(1 for shape in pattern if shape != "WORD")
        score = (len(positions) * 10) + (unique_labels * 2) + non_word_count

        if score > best_score:
            best_score = score
            best_pattern = pattern
            best_positions = positions

    if best_pattern is None or best_positions is None:
        return None

    start_token_idx = min(best_positions)

    # Convert token index to approximate character position
    running_pos = 0
    for idx, token in enumerate(tokens):
        if idx == start_token_idx:
            return running_pos
        running_pos += len(token) + 1

    return None


def find_previous_sentence_boundary(text, pos):
    """
    Find the last reasonable sentence boundary before the given position.
    """
    if pos is None or pos <= 0:
        return None

    boundary_chars = {'.', '!', '?', ';', '\n'}

    for i in range(min(pos, len(text)) - 1, -1, -1):
        if text[i] in boundary_chars:
            return i + 1

    return None


def truncate_repeating_structure(text):
    """
    If text is longer than MIN_LENGTH_FOR_PATTERN_CHECK and contains repeated
    token-shape patterns, truncate at the previous sentence boundary.
    """
    if len(text) <= MIN_LENGTH_FOR_PATTERN_CHECK:
        return text, False

    repeating_start = find_repeating_shape_start(text)
    if repeating_start is None:
        return text, False

    cut_pos = find_previous_sentence_boundary(text, repeating_start)
    if cut_pos is None:
        return text, False

    truncated_text = text[:cut_pos].rstrip() + TRUNCATION_LABEL
    return truncated_text, True


def should_skip_all_study_fields(file_path):
    """
    Return True if any field name starting with 'study' appears more than once
    in the file. In that case, all study* fields should be excluded.
    """
    seen_study_fields = set()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('=', 1)
            if len(parts) < 2:
                continue

            field_name = parts[0].strip().lower()

            if field_name.startswith("study"):
                if field_name in seen_study_fields:
                    return True
                seen_study_fields.add(field_name)

    return False


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

        skip_all_study_fields = should_skip_all_study_fields(file_path)

        if skip_all_study_fields:
            with open(log_file_path, 'a') as log_file:
                log_file.write(
                    f"Duplicate study field name detected in {os.path.basename(file_path)}; "
                    f"all study* fields will be skipped.\n"
                )

        new_lines = []

        with open(file_path, 'r') as file:
            for line in file:
                original_line = line  # Keep the original line for logging purposes
                line_lower = line.lower().strip()
                rejection_endings = (
                    "=",
                    "nan",
                    "not applicable",
                    "missing",
                    " na",
                    " na\n",
                    "not collected",
                    "unknown",
                    "unspecified",
                    "not provided"
                )

                # Check if the line should be rejected
                if line_lower.endswith(rejection_endings) or line_lower.startswith(("experiment", "run")):
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Rejected line: {line}")
                    continue

                # Split line on first equals sign, if it exists
                parts = line.split('=', 1)
                field_name = parts[0].strip() if len(parts) > 1 else ''
                field_name_lower = field_name.lower()
                prefix = parts[0] + '=' if len(parts) > 1 else ''
                line_to_process = parts[-1]

                # If duplicate study field names exist anywhere in this file,
                # skip all study* fields
                if skip_all_study_fields and field_name_lower.startswith("study"):
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Skipped study field: {field_name}\n")
                    continue

                # Detect repeating structured content in long fields and truncate
                line_to_process, was_truncated = truncate_repeating_structure(line_to_process)
                if was_truncated:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(
                            f"Truncated repeating structured content in field "
                            f"'{field_name}' in file {os.path.basename(file_path)}\n"
                        )

                for word in line_to_process.split():
                    numeric_part = ''.join(filter(str.isdigit, word))
                    word_lower = word.lower()  # Convert word to lower case for matching
                    if numeric_part in modified_ontology_dict:
                        for namespace, desc in modified_ontology_dict[numeric_part].items():
                            if namespace in word_lower:
                                replacement = f"'{desc}'"
                                line_to_process = line_to_process.replace(word, replacement)
                                with open(log_file_path, 'a') as log_file:
                                    log_file.write(
                                        f"Converting '{word}' in line "
                                        f"'{original_line.strip()}' to '{replacement}'\n"
                                    )
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

    parser = argparse.ArgumentParser(description='Clean metadata and translate ontology codes to labels')
    parser.add_argument('--path_to_dir', default='.', help='Path to working directory (default: current directory)')
    parser.add_argument('--ontology_dict', required=True, help='Ontology dictionary file (.pkl)')
    parser.add_argument('--metadata_dirs', required=True, help='Directory name(s) containing metadata files')
    parser.add_argument('--max_processes', type=int, default=1, help='Number of parallel processes (default: 1)')

    args = parser.parse_args()

    ontology_dict_path = os.path.join(os.path.expanduser(args.path_to_dir), args.ontology_dict)
    with open(ontology_dict_path, 'rb') as f:
        ontology_dict = pickle.load(f)

    # Modify the ontology dictionary
    modified_ontology_dict = modify_ontology_dict(ontology_dict)

    base_dir = os.path.join(os.path.expanduser(args.path_to_dir), args.metadata_dirs)
    base_log_file_path = os.path.join(base_dir, "log_clean_and_envo_translate")

    all_dir_names = [dir_name for dir_name in os.listdir(base_dir) if dir_name.startswith("dir_")]
    max_processes = min(args.max_processes, multiprocessing.cpu_count())

    for i in range(0, len(all_dir_names), max_processes):
        processes = []
        for dir_name in all_dir_names[i:i + max_processes]:
            dir_path = os.path.join(base_dir, dir_name)
            p = multiprocessing.Process(
                target=process_directory,
                args=(dir_path, modified_ontology_dict, base_log_file_path)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()  # Wait for the current batch of processes to complete before starting the next batch

    end_time = time.time()
    print(f"Script executed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
    

# old code below (replaced with code above on 2026.04.14) to include: 
# 1. how to avoid taking along metadata that contains info from different studies: if the field starts with 'study' and the same field name is again seen later, skip all the 'study' fields. don t take these along to the clean data
# 2. how to avoid taking abundance tables along: if field has > 1000 characters, convert tokens to token shapes and detect repeating shape patterns. if detected, truncate at previous sentence boundary.
# =============================================================================
# import os
# import pickle
# import argparse
# import multiprocessing
# import time
# from datetime import datetime
# from multiprocessing import Process, cpu_count
# 
# 
# ##############################################################################
# # # before running script, change ulimit in the session: 
# # # $ ulimit -n 200000 <-- it's an estimation derived from: 
# # # 40 (dirs and cpus at a time) * 3800 (files per dir) = 152000 --> round up: 200000
# # # then run on local: 
# # python ~/github/metadata_mining/scripts/clean_and_envo_translate.py \
# #     --path_to_dir "~/MicrobeAtlasProject" \
# #     --ontology_dict "ontologies_dict.pkl" \
# #     --metadata_dirs "sample_info_split_dirs" \ 
# #     --max_processes 8
# # # or on atlas: --max_processes 40
# ##############################################################################
# 
# 
# def modify_ontology_dict(ontology_dict):
#     modified_dict = {}
#     for term, desc in ontology_dict.items():
#         namespace, _, numeric_part = term.partition('_')
#         namespace = namespace.lower()  # Convert namespace to lower case
#         if numeric_part not in modified_dict:
#             modified_dict[numeric_part] = {}
#         modified_dict[numeric_part][namespace] = desc
#     return modified_dict
# 
# 
# def process_directory(dir_path, modified_ontology_dict, base_log_file_path):
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     dir_name = os.path.basename(dir_path)
#     log_file_path = f"{base_log_file_path}_log_{dir_name}_{current_time}.txt"
# 
#     for file_name in os.listdir(dir_path):
#         if file_name.endswith(".txt") and not file_name.endswith("_clean.txt"):
#             process_file(os.path.join(dir_path, file_name), modified_ontology_dict, log_file_path)
# 
# 
# def process_file(file_path, modified_ontology_dict, log_file_path):
#     try:
#         with open(log_file_path, 'a') as log_file:
#             log_file.write(f"Processing file {os.path.basename(file_path)}...\n")
# 
#         new_lines = []
#         with open(file_path, 'r') as file:
#             for line in file:
#                 original_line = line  # Keep the original line for logging purposes
#                 line_lower = line.lower().strip()
#                 rejection_endings = ("=", "nan", "not applicable", "missing", " na", " na\n", "not collected", "unknown", "unspecified", "not provided")
# 
#                 # Check if the line should be rejected
#                 if line_lower.endswith(rejection_endings) or line_lower.startswith(("experiment", "run")):
#                     with open(log_file_path, 'a') as log_file:
#                         log_file.write(f"Rejected line: {line}")
#                     continue  # Skip the rest of the processing for this line and do not add it to new_lines
# 
#                 # Split line on first equals sign, if it exists
#                 parts = line.split('=', 1)
#                 prefix = parts[0] + '=' if len(parts) > 1 else ''
#                 line_to_process = parts[-1]
# 
#                 for word in line_to_process.split():
#                     numeric_part = ''.join(filter(str.isdigit, word))
#                     word_lower = word.lower()  # Convert word to lower case for matching
#                     if numeric_part in modified_ontology_dict:
#                         for namespace, desc in modified_ontology_dict[numeric_part].items():
#                             if namespace in word_lower:
#                                 replacement = f"'{desc}'"
#                                 line_to_process = line_to_process.replace(word, replacement)
#                                 with open(log_file_path, 'a') as log_file:
#                                     log_file.write(f"Converting '{word}' in line '{original_line.strip()}' to '{replacement}'\n")
#                                 break  # Stop checking other namespaces for this word
# 
#                 new_lines.append(prefix + line_to_process)
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
# 
#     parser = argparse.ArgumentParser(description='Clean metadata and translate ontology codes to labels')
#     parser.add_argument('--path_to_dir', default='.', help='Path to working directory (default: current directory)')
#     parser.add_argument('--ontology_dict', required=True, help='Ontology dictionary file (.pkl)')
#     parser.add_argument('--metadata_dirs', required=True, help='Directory name(s) containing metadata files')
#     parser.add_argument('--max_processes', type=int, default=1, help='Number of parallel processes (default: 1)')
# 
#     args = parser.parse_args()
# 
#     ontology_dict_path = os.path.join(os.path.expanduser(args.path_to_dir), args.ontology_dict)
#     with open(ontology_dict_path, 'rb') as f:
#         ontology_dict = pickle.load(f)
# 
#     # Modify the ontology dictionary
#     modified_ontology_dict = modify_ontology_dict(ontology_dict)
# 
#     base_dir = os.path.join(os.path.expanduser(args.path_to_dir), args.metadata_dirs)
#     base_log_file_path = os.path.join(base_dir, "log_clean_and_envo_translate")
# 
#     all_dir_names = [dir_name for dir_name in os.listdir(base_dir) if dir_name.startswith("dir_")]
#     max_processes = min(args.max_processes, multiprocessing.cpu_count())
# 
#     for i in range(0, len(all_dir_names), max_processes):
#         processes = []
#         for dir_name in all_dir_names[i:i + max_processes]:
#             dir_path = os.path.join(base_dir, dir_name)
#             p = multiprocessing.Process(target=process_directory, args=(dir_path, modified_ontology_dict, base_log_file_path))
#             processes.append(p)
#             p.start()
# 
#         for p in processes:
#             p.join()  # Wait for the current batch of processes to complete before starting the next batch
# 
#     end_time = time.time()
#     print(f"Script executed in {end_time - start_time:.2f} seconds")
# 
# 
# if __name__ == "__main__":
#     main()
# =============================================================================





