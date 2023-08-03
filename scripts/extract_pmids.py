#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:10:34 2023

@author: dgaio
"""

# run as: 
# python github/metadata_mining/scripts/extract_pmids.py --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_csv '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid.csv' --plot --figure_path '~/cloudstor/Gaio/MicrobeAtlasProject/pmids_NaN_vs_nonNaN.pdf'


import os
import re
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def find_pmids_from_large_file(file_path):
    pmid_dict = {}
    pmid_pattern = re.compile(r"(PMID|pmid)\D*(\d{8})", re.IGNORECASE)
    
    start_time = time.time()
    sample_counter = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sample_name = ''
    for line in lines:
        if line.startswith('>'):
            sample_name = line.replace('>', '').strip()
            pmid_dict[sample_name] = []
            sample_counter += 1

            if sample_counter % 200000 == 0:  # placed inside the condition that checks if a new sample has been found
                elapsed_time = time.time() - start_time
                print(f"Processed {sample_counter} samples in {elapsed_time:.2f} seconds "  
                      f"({sample_counter / elapsed_time:.2f} samples/second)")
        else:
            match = pmid_pattern.search(line)
            if match:
                pmid_dict[sample_name].append(match.group())

    return pmid_dict


def dict_to_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample', 'pmid', 'pmid_digits'])

        for key, values in dictionary.items():
            if values:
                pmid = '&'.join(values)
                pmid_digits = re.findall(r'\d+', pmid)
                first_digits = pmid_digits[0] if pmid_digits else ''
                writer.writerow([key, pmid, first_digits])
            else:
                writer.writerow([key, '', ''])

                
def plot_pmid_info(filename, figure_path=None):
    data = pd.read_csv(filename)
    non_na_count = data['pmid_digits'].notna().sum()
    na_count = data['pmid_digits'].isna().sum()

    labels = ['Non-NA', 'NA']
    counts = [non_na_count, na_count]

    plt.bar(labels, counts)
    plt.xlabel('Column "pmid_digits"')
    plt.ylabel('Count')
    plt.title('Count of Non-NA Values in Column "pmid_digits"')

    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    if figure_path:
        plt.savefig(figure_path, format='pdf')
    else:
        plt.show()


parser = argparse.ArgumentParser(description='Find PMIDs in the large file.')
parser.add_argument('--large_file', type=str, required=True, help='Path to the large input file')
parser.add_argument('--output_csv', type=str, required=True, help='Output csv file')
parser.add_argument('--plot', action='store_true', help='Plot the count of non-NA values')
parser.add_argument('--figure_path', type=str, help='Optional path to save the histogram figure')
args = parser.parse_args()

large_file_path = os.path.expanduser(args.large_file)
output_csv = os.path.expanduser(args.output_csv)
figure_path = os.path.expanduser(args.figure_path) if args.figure_path else None

pmid_dict = find_pmids_from_large_file(large_file_path)
dict_to_csv(pmid_dict, output_csv)

if args.plot:
    plot_pmid_info(output_csv, figure_path=figure_path)










# =============================================================================
# import pandas as pd
# import numpy as np
# 
# def find_pmids_from_large_file(file_path):
#     pmid_dict = {}
#     pmid_pattern = re.compile(r"(PMID|pmid)\D*(\d+)", re.IGNORECASE)
#     
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
# 
#     sample_name = ''
#     for line in lines:
#         if line.startswith('>'):
#             sample_name = line.replace('>', '').strip()
#             pmid_dict[sample_name] = []
#         else:
#             match = pmid_pattern.search(line)
#             if match:
#                 pmid_dict[sample_name].append(match.group(2))  # get only the digits of PMID
#                 
#     return pmid_dict
# 
# def compare_with_shell_output(python_dict, shell_file):
#     # Load shell output into a pandas Series
#     shell_output = pd.read_csv(shell_file, header=None, squeeze=True)
#     
#     # Flatten python_dict values into a list
#     python_output = [pmid for pmids in python_dict.values() for pmid in pmids]
#     python_output = pd.Series(python_output)
#     
#     # Compare
#     not_in_python = shell_output.loc[~shell_output.isin(python_output)]
#     not_in_shell = python_output.loc[~python_output.isin(shell_output)]
#     
#     print("PMIDs found in shell output but not in Python output:\n", not_in_python)
#     print("PMIDs found in Python output but not in shell output:\n", not_in_shell)
# 
# # Usage
# pmid_dict = find_pmids_from_large_file("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info")
# compare_with_shell_output(pmid_dict, "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/get_pmids_test.txt")
# =============================================================================

