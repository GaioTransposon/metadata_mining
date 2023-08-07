#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:28:27 2023

@author: dgaio
"""

# run as: 
# python github/metadata_mining/scripts/extract_doi.py --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_csv '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_doi.csv' 

import os
import re
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from Bio import Entrez
import itertools
import numpy as np
from collections.abc import Iterable

def find_dois_from_large_file(file_path):
    doi_dict = {}
    doi_pattern = re.compile(r"\bdoi:10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
    
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
    return unique_dois

def dois_to_pmids(dois):
    query = " OR ".join(dois)
    Entrez.email = "daniela.gaio@mls.uzh.ch"
    handle = Entrez.esearch(db="pubmed", term=query)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def map_dois_to_pmids(dois):
    dois = [doi.lower() for doi in dois] # Convert to lowercase
    dois = list(set(dois)) # Convert to list and remove duplicates
    doi_to_pmid = {}
    total_dois = len(dois)
    batch_size = 200
    start_time = time.time()
    
    for idx in range(0, total_dois, batch_size):
        batch_dois = dois[idx:idx+batch_size]
        for doi in batch_dois:
            pmids = dois_to_pmids([doi])
            if pmids:
                doi_to_pmid[doi] = pmids[0]
        
        processed_dois = idx + len(batch_dois)
        elapsed_time = time.time() - start_time
        remaining_time = (total_dois - processed_dois) * (elapsed_time / processed_dois)
        print(f"Processed {processed_dois} DOIs out of {total_dois} at {processed_dois / elapsed_time:.2f} DOIs/second. Expected time remaining: {remaining_time:.2f} seconds.")
        
        time.sleep(3) # Sleep for 3 seconds

    return doi_to_pmid

# Define a function to apply to the "dois" column to generate "pmid_digits"
def doi_to_pmid(doilst):
    pmid_digits = [doi_to_pmid_mapping[doi.lower()] for doi in doilst if doi.lower() in doi_to_pmid_mapping]
    return pmid_digits if pmid_digits else np.nan

def join_if_iterable(value):
    if isinstance(value, Iterable) and not isinstance(value, str):
        return '; '.join(value)
    return value

parser = argparse.ArgumentParser(description='Find PMIDs in the large file.')
parser.add_argument('--large_file', type=str, required=True, help='Path to the large input file')
parser.add_argument('--output_csv', type=str, required=True, help='Output csv file')
args = parser.parse_args()

large_file_path = os.path.expanduser(args.large_file)
output_csv = os.path.expanduser(args.output_csv)


doi_dict = find_dois_from_large_file(large_file_path)

# for testing purposes
# d = dict(itertools.islice(doi_dict.items(), 100))
d = doi_dict


# make a list of uniq doi, so you only have to request those: 
unique_dois = extract_unique_dois(d)
doi_to_pmid_mapping = map_dois_to_pmids(unique_dois)


# Convert to DataFrame
df = pd.DataFrame(list(d.items()), columns=['sample', 'dois'])


# Apply the function to create the "pmid_digits" column
df['pmid_digits'] = df['dois'].apply(doi_to_pmid)


# remove square brackets from both dois and plid_digits columns: 
df['dois'] = df['dois'].apply(join_if_iterable)
df['pmid_digits'] = df['pmid_digits'].apply(join_if_iterable)


# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)




# save csv instead as a dictionary item 











