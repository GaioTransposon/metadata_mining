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


def dois_to_pmids(dois):
    dois = [doi.lower() for doi in dois] # Convert to lowercase
    dois = list(set(dois)) # Convert to list and remove duplicates
    doi_to_pmid = {}
    total_dois = len(dois)
    batch_size = 200
    start_time = time.time()
    
    for idx in range(0, total_dois, batch_size):
        batch_dois = dois[idx:idx+batch_size]
        query = " OR ".join([f"{doi}[DOI]" for doi in batch_dois])

        Entrez.email = "daniela.gaio@mls.uzh.ch"
        handle = Entrez.esearch(db="pubmed", term=query)
        record = Entrez.read(handle)
        handle.close()

        # Map the DOIs to PMIDs
        for doi, pmid in zip(batch_dois, record["IdList"]):
            doi_to_pmid[doi] = pmid if pmid else None
        
        processed_dois = idx + len(batch_dois)
        elapsed_time = time.time() - start_time
        remaining_time = (total_dois - processed_dois) * (elapsed_time / processed_dois)
        print(f"Processed {processed_dois} DOIs out of {total_dois} at {processed_dois / elapsed_time:.2f} DOIs/second. Expected time remaining: {remaining_time:.2f} seconds.")
        
        time.sleep(3) # Sleep for 3 seconds

    return doi_to_pmid


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

my_doi_to_pmid = dois_to_pmids(unique_dois)

# if the doi in my_doi_to_pmid has got a corresponding pmid, then this pmid will be added to the list of the original dictionary at the respective place (not replaced, but added)
for key, values in doi_dict.items():
    for value in values:
        if value in my_doi_to_pmid:
            doi_dict[key].append(my_doi_to_pmid[value])

# remove keys that have no info (neither doi nor pmid):
doi_dict = {key: value for key, value in doi_dict.items() if value}

save_to_json(doi_dict, output_file)



# Print some stats: 
doi_count = 0
pmid_count = 0
unique_dois = set()
unique_pmids = set()

for values in doi_dict.values():
    for value in values:
        if value.startswith('doi:'):
            unique_dois.add(value)
        elif value.isdigit():
            unique_pmids.add(value)

doi_count = len(unique_dois)
pmid_count = len(unique_pmids)

print(f"Tot number of unique DOIs is {doi_count}")
print(f"Tot number of unique PMIDs is {pmid_count}")
print(f"among {len(doi_dict)} samples")




