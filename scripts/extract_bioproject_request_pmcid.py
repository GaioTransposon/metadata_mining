#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:40:36 2023

@author: dgaio
"""


# run as: 
# python github/metadata_mining/scripts/extract_bioproject_request_pmcid.py --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_bioproject' 
# takes about 8h ! 


import os
import re
import time
import argparse
from Bio import Entrez
import json
import itertools

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



def extract_unique_bioprojects(dictionary):
    unique_bioprojects = set()
    for bioproject in dictionary.values():
        unique_bioprojects.update(bioproject)
    return list(unique_bioprojects)



def fetch_pmcids(bioprojects_set):
    # Convert the set to a list
    bioprojects_list = list(bioprojects_set)

    # Set the batch size
    batch_size = 200
    total_batches = (len(bioprojects_list) + batch_size - 1) // batch_size

    bioprojects_pmc_dic = {}

    # Make sure to include your email address, as NCBI requires it for tracking purposes
    Entrez.email = "daniela.gaio@mls.uzh.ch"

    for batch_num in range(total_batches):
        start_time = time.time()

        # Determine the start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(bioprojects_list))

        # Process the bioprojects in this batch
        for i in range(start_idx, end_idx):
            bioproject = bioprojects_list[i]
            print(f"Processing {bioproject}...")

            # Fetch the related articles for the BioProject ID
            query = f"{bioproject}[BioProject]"
            handle = Entrez.esearch(db="pmc", term=query)    
            record = Entrez.read(handle)
            handle.close()

            # pmc_ids = record['IdList'] # old
            pmc_ids = ['PMC' + pmc_id for pmc_id in record['IdList']]

            bioprojects_pmc_dic[bioproject] = pmc_ids

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
    return bioprojects_pmc_dic



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


bioprojects_dict = find_bioprojects_from_large_file(large_file_path)

unique_bioprojects = extract_unique_bioprojects(bioprojects_dict)

# # for testing purpses
unique_bioprojects = set(itertools.islice(unique_bioprojects, 600))


bioprojects_pmcid_dic = fetch_pmcids(unique_bioprojects)


# I think what it really takes to stay in science is naivity to the point of being overly hopeful and optimistic. Dumb. In other words. 



# Create last dictionary
a = bioprojects_dict
b = bioprojects_pmcid_dic

new_dic = {}

for key, bioprojects in a.items():
    # Iterating through the list of bioprojects
    for bioproject in bioprojects:
        # Getting the PMCIDs for each bioproject
        pmcids = b.get(bioproject, [])
        # If PMCIDs exist, add them to the new dictionary
        if pmcids:
            new_dic[key] = pmcids


save_to_json(new_dic, output_file)




# Stats:
unique_bioprojectss = set()
unique_pmcidss = set()

# Iterate through bioprojects_dict to collect unique BioProjects
for bioprojects in bioprojects_dict.values():
    for bioproject in bioprojects:
        if bioproject.startswith('PRJNA'):
            unique_bioprojectss.add(bioproject)

# Iterate through bioprojects_pmcid_dic to collect unique PMCIDs
for pmcids in bioprojects_pmcid_dic.values():
    unique_pmcidss.update(pmcids)  # Using update to add all PMCIDs in the list to the set

# Printing the results
print(f"Total number of samples with BioProjects: {len(bioprojects_dict)}")
print(f" of which unique BioProjects: {len(unique_bioprojectss)}")
print(f"Number of PMCIDs found: {len(new_dic)}")
print(f" of which unique PMCIDs: {len(unique_pmcidss)}")



