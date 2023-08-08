#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:40:36 2023

@author: dgaio
"""


# run as: 
# python github/metadata_mining/scripts/extract_bioproject_request_pmid.py --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_bioproject' 

import os
import re
import time
import argparse
from Bio import Entrez
import json

def find_bioprojects_from_large_file(file_path):
    bioproject_dict = {}
    bioproject_pattern = re.compile(r"PRJNA\s?\d+|\bbioproject[:/\s]\s*(\d+)\b", re.IGNORECASE)

    start_time = time.time()
    sample_counter = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sample_name = ''
    for line in lines:
        if line.startswith('>'):
            sample_name = line.replace('>', '').strip()
            bioproject_dict[sample_name] = []
            sample_counter += 1

            if sample_counter % 200000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {sample_counter} samples in {elapsed_time:.2f} seconds "
                      f"({sample_counter / elapsed_time:.2f} samples/second)")
        else:
            matches = bioproject_pattern.findall(line)
            for match in matches:
                match = match.replace(" ", "")  # Remove spaces if any
                if match.startswith('PRJNA'):
                    standardized_bioproject_id = match
                else:
                    standardized_bioproject_id = 'PRJNA' + match
                
                bioproject_dict[sample_name].append(standardized_bioproject_id)  # get the entire Bioproject ID

    return bioproject_dict




# for testing purposes 
large_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info'


bioprojects_dict = find_bioprojects_from_large_file(large_file_path)







def extract_unique_bioprojects(dictionary):
    unique_bioprojects = set()
    for bioproject in dictionary.values():
        unique_bioprojects.update(bioproject)
    return list(unique_bioprojects)



unique_bioprojects = extract_unique_bioprojects(bioprojects_dict)
len(unique_bioprojects)






def bioprojects_to_pmids(bioprojects):
    bioprojects = list(set(bioprojects)) # Remove duplicates
    bioproject_to_pmid = {}
    total_bioprojects = len(bioprojects)
    batch_size = 200
    start_time = time.time()
    
    for idx in range(0, total_bioprojects, batch_size):
        batch_bioprojects = bioprojects[idx:idx+batch_size]
        query = " OR ".join([f"{bioproject}[BioProject]" for bioproject in batch_bioprojects])

        Entrez.email = "daniela.gaio@mls.uzh.ch"
        handle = Entrez.esearch(db="bioproject", term=query)
        record = Entrez.read(handle)
        handle.close()

        # Map the BioProjects to PMIDs
        for bioproject, pmid in zip(batch_bioprojects, record["IdList"]):
            bioproject_to_pmid[bioproject] = pmid if pmid else None
        
        processed_bioprojects = idx + len(batch_bioprojects)
        elapsed_time = time.time() - start_time
        remaining_time = (total_bioprojects - processed_bioprojects) * (elapsed_time / processed_bioprojects)
        print(f"Processed {processed_bioprojects} BioProjects out of {total_bioprojects} at {processed_bioprojects / elapsed_time:.2f} BioProjects/second. Expected time remaining: {remaining_time:.2f} seconds.")
        
        time.sleep(3) # Sleep for 3 seconds

    return bioproject_to_pmid






my_bioprojects_to_pmids = bioprojects_to_pmids(unique_bioprojects)





# I think what it really takes to stay in science is naivity to the point of being overly hopeful and optimistic. Dumb. In other words. 







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
# output_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_bioproject' 


doi_dict = find_bioprojects_from_large_file(large_file_path)

unique_bioprojects = extract_unique_bioprojects(doi_dict)

my_bioproject_to_pmid = bioprojects_to_pmids(unique_bioprojects)

# if the bioproject in my_bioproject_to_pmid has got a corresponding pmid, then this pmid will be added to the list of the original dictionary at the respective place (not replaced, but added)
for key, values in bioprojects_dict.items():
    for value in values:
        if value in my_bioproject_to_pmid:
            bioprojects_dict[key].append(my_bioproject_to_pmid[value])

# remove keys that have no info (neither doi nor pmid):
bioprojects_dict = {key: value for key, value in bioprojects_dict.items() if value}

save_to_json(bioprojects_dict, output_file)



# Print some stats: 
bioproject_count = 0
pmid_count = 0
unique_bioprojects = set()
unique_pmids = set()

for values in bioprojects_dict.values():
    for value in values:
        if value.startswith('PRJNA:'):
            unique_bioprojects.add(value)
        elif value.isdigit():
            unique_pmids.add(value)

bioproject_count = len(unique_bioprojects)
pmid_count = len(unique_pmids)

print(f"Tot number of unique bioprojects is {bioproject_count}")
print(f"Tot number of unique PMIDs is {pmid_count}")
print(f"among {len(bioprojects_dict)} samples")

