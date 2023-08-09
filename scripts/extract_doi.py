#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:28:27 2023

@author: dgaio
"""

# run as: 
# python github/metadata_mining/scripts/extract_doi.py --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_doi' 

import os
import re
import time
import argparse
from Bio import Entrez
import json

def find_dois_from_large_file(file_path):
    doi_dict = {}
    doi_pattern = re.compile(r"\bdoi:[^\s]+\b", re.IGNORECASE)

    
    start_time = time.time()
    sample_counter = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sample_name = ''
    for line in lines:
        if line.startswith('>'):
            sample_name = line.replace('>', '').strip()
            doi_dict[sample_name] = []
            sample_counter += 1

            if sample_counter % 200000 == 0:  # placed inside the condition that checks if a new sample has been found
                elapsed_time = time.time() - start_time
                print(f"Processed {sample_counter} samples in {elapsed_time:.2f} seconds "
                      f"({sample_counter / elapsed_time:.2f} samples/second)")
        else:
            matches = doi_pattern.findall(line)
            for match in matches:
                doi_dict[sample_name].append(match)  # get the entire DOI

    return doi_dict


def extract_unique_dois(dictionary):
    unique_dois = set()
    for dois in dictionary.values():
        unique_dois.update(dois)
    return list(unique_dois)


def fetch_pmids(dois_set):
    # Convert the set to a list
    dois_list = list(dois_set)

    # Set the batch size
    batch_size = 200
    total_batches = (len(dois_list) + batch_size - 1) // batch_size

    dois_pmid_dic = {}

    # Make sure to include your email address, as NCBI requires it for tracking purposes
    Entrez.email = "daniela.gaio@mls.uzh.ch"

    for batch_num in range(total_batches):
        start_time = time.time()

        # Determine the start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(dois_list))

        # Process the DOIs in this batch
        for i in range(start_idx, end_idx):
            doi = dois_list[i]
            print(f"Processing {doi}...")

            # Fetch the related articles for the DOI
            query = f"{doi}[DOI]"
            handle = Entrez.esearch(db="pubmed", term=query)
            record = Entrez.read(handle)            
            handle.close()
            
            pmid_ids = record['IdList']
            
            if pmid_ids:
                print(f"Found {len(pmid_ids)} item(s) for {doi}.")
                dois_pmid_dic[doi] = pmid_ids
            else:
                print(f"No items found for {doi}.\n")


        # Sleep for 3 seconds
        time.sleep(3)
        
        # Calculate and print the time taken for this batch
        batch_time = time.time() - start_time
        print(f"Batch {batch_num + 1}/{total_batches} took {batch_time:.2f} seconds.")

        # Estimate the remaining time
        remaining_batches = total_batches - batch_num - 1
        remaining_time = batch_time * remaining_batches
        print(f"Estimated remaining time: {remaining_time / 60:.2f} minutes.")

    # Return the result
    return dois_pmid_dic



def save_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)


parser = argparse.ArgumentParser(description='Find PMIDs in the large file.')
parser.add_argument('--large_file', type=str, required=True, help='Path to the large input file')
parser.add_argument('--output_file', type=str, required=True, help='Output file')
args = parser.parse_args()

large_file_path = os.path.expanduser(args.large_file)
output_file = os.path.expanduser(args.output_file)


# # for testing purposes 
# large_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info'
# output_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_doi' 


doi_dict = find_dois_from_large_file(large_file_path)

unique_dois = extract_unique_dois(doi_dict)



unique_dois_small = unique_dois[1:30]

my_doi_to_pmid = fetch_pmids(unique_dois_small)



# if the doi in my_doi_to_pmid has got a corresponding pmid, then this pmid will be added to the list of the original dictionary at the respective place (not replaced, but added)
new_dict = {}
for key, dois in doi_dict.items():
    for doi in dois:
        doi_value = doi if isinstance(doi, str) else doi[0]
        if doi_value in my_doi_to_pmid:
            new_dict[key] = my_doi_to_pmid[doi_value]


# remove keys that have no info (neither doi nor pmid):
new_dict = {key: value for key, value in new_dict.items() if value}

save_to_json(new_dict, output_file)



# Print some stats: 
pmid_count = 0
unique_pmids = set()

for values in new_dict.values():
    for value in values:
        unique_pmids.add(value)
            
pmid_count = len(unique_pmids)

print(f"Tot number of unique dois is {len(unique_dois)}")
print(f"Tot number of unique PMIDs is {pmid_count}")
print(f"among {len(new_dict)} samples")








