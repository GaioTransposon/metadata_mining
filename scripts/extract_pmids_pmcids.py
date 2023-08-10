#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:10:34 2023

@author: dgaio
"""

# # run as: 
# python ~/github/metadata_mining/scripts/extract_pmids_pmcids.py  \
#         --large_file_subset '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_subset' \
#             --output_file_pmid '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid' \
#                 --output_file_pmcid '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmcid'



import os
import re
import time
import argparse
import json



def find_pmids_and_pmcids_from_large_file(file_path):
    pmid_dict = {}
    pmcid_dict = {}
    
    pmid_pattern = re.compile(r"(PMID|pmid)\D*([\d]+)", re.IGNORECASE)
    pmcid_pattern = re.compile(r"\bPMC(\d+)\b")
    
    start_time = time.time()
    sample_counter = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    

    sample_name = ''
    for line in lines:
        if line.startswith('>'):
            sample_name = line.replace('>', '').strip()
            
            pmid_dict[sample_name] = []
            pmcid_dict[sample_name] = []
            
            sample_counter += 1

            if sample_counter % 200000 == 0:  # placed inside the condition that checks if a new sample has been found
                elapsed_time = time.time() - start_time
                print(f"Processed {sample_counter} samples in {elapsed_time:.2f} seconds "  
                      f"({sample_counter / elapsed_time:.2f} samples/second)")
        else:
            
            matches_pmid = pmid_pattern.findall(line)
            for match in matches_pmid:
                pmid_dict[sample_name].append(match[1])  # get only the digits of PMID
            
            matches_pmcid = pmcid_pattern.findall(line)
            for match in matches_pmcid:
                match='PMC'+match
                pmcid_dict[sample_name].append(match)  


    return pmid_dict, pmcid_dict



def filter_non_empty_values(original_dict):
    return {key: list(set(value)) for key, value in original_dict.items() if value}



def save_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)
        

parser = argparse.ArgumentParser(description='Find PMIDs in the large file.')
parser.add_argument('--large_file_subset', type=str, required=True, help='Path to the large input file subset')
parser.add_argument('--output_file_pmid', type=str, required=True, help='Output file')
parser.add_argument('--output_file_pmcid', type=str, required=True, help='Output file')
args = parser.parse_args()

large_file_subset = os.path.expanduser(args.large_file_subset)
output_file_pmid = os.path.expanduser(args.output_file_pmid)
output_file_pmcid = os.path.expanduser(args.output_file_pmcid)


# for testing purposes: 
# large_file_subset = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info'
# output_file_pmid = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid'
# output_file_pmcid = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmcid'





pmid_dict,pmcid_dict = find_pmids_and_pmcids_from_large_file(large_file_subset)



my_pmid_dict = filter_non_empty_values(pmid_dict)

my_pmcid_dict = filter_non_empty_values(pmcid_dict)


save_to_json(my_pmid_dict, output_file_pmid)
save_to_json(my_pmcid_dict, output_file_pmcid)


# Print some stats: 
all_values = [value for values in my_pmid_dict.values() for value in values]
unique_values_count = len(set(all_values))
n_samples = len(my_pmid_dict)
print(f"Tot number of uniq pmids among {n_samples} samples is {unique_values_count}")

# Print some stats: 
all_values = [value for values in my_pmcid_dict.values() for value in values]
unique_values_count = len(set(all_values))
n_samples = len(my_pmcid_dict)
print(f"Tot number of uniq pmcids among {n_samples} samples is {unique_values_count}")











