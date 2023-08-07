#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:10:34 2023

@author: dgaio
"""

# run as: 
# python github/metadata_mining/scripts/extract_pmids.py --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid' 


import os
import re
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json


def find_pmids_from_large_file(file_path):
    pmid_dict = {}
    pmid_pattern = re.compile(r"(PMID|pmid)\D*([\d]+)", re.IGNORECASE)
    
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
            matches = pmid_pattern.findall(line)
            for match in matches:
                pmid_dict[sample_name].append(match[1])  # get only the digits of PMID

    return pmid_dict


def transform_dict(original_dict):
    new_dict = {}
    for key, values in original_dict.items():
        if values:
            pmid = '&'.join(values)
            pmid_digits = re.findall(r'\d+', pmid)
            unique_pmid_digits = list(set(pmid_digits))  # Keep only unique values
            new_dict[key] = unique_pmid_digits
    return new_dict


def save_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)
        

parser = argparse.ArgumentParser(description='Find PMIDs in the large file.')
parser.add_argument('--large_file', type=str, required=True, help='Path to the large input file')
parser.add_argument('--output_file', type=str, required=True, help='Output csv file')
args = parser.parse_args()

large_file_path = os.path.expanduser(args.large_file)
output_file = os.path.expanduser(args.output_file)


# for testing purposes: 
# large_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info'


pmid_dict = find_pmids_from_large_file(large_file_path)



my_pmid_dict = transform_dict(pmid_dict)
print(my_pmid_dict)  




save_to_json(my_pmid_dict, output_file)















