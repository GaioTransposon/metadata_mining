#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:49:47 2023

@author: dgaio
"""


# # run as: 
# python ~/github/metadata_mining/scripts/join_all.py  \
#         --work_dir '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/' \
#             --biomes_df 'samples_biomes' \
#                 --pmids_dict_path 'sample.info_pmid' \
#                     --pmcids_dict_path 'sample.info_pmcid' \
#                         --dois_pmids_dict_path 'sample.info_doi' \
#                             --bioprojects_pmcid_dict_path 'sample.info_bioproject' \
#                                 --output_file 'sample.info_biome_pmid.csv' \
#                                     --figure 'sample.info_biome_pmid.pdf' \
## Code takes ca 40' to run 

import os
import time
import argparse
from Bio import Entrez
import json
import xml.etree.ElementTree as ET
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def merge_dicts(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            result[key] = list(set(result[key] + value))
        else:
            result[key] = value
    return result


def unique_values(dictionary):
    unique_vals = set()
    for values in dictionary.values():
        if isinstance(values, list):
            unique_vals.update(values)
        else:
            unique_vals.add(values)
    return list(unique_vals)


def from_pmcids_to_pmids(pmcids):
    # Convert the input to a list (if it's not)
    pmcids_list = list(pmcids)

    n=0
    # Set the batch size
    batch_size = 200
    total_batches = (len(pmcids_list) + batch_size - 1) // batch_size

    pmid_dict = {}

    # Provide the required email
    Entrez.email = "daniela.gaio@mls.uzh.ch"

    for batch_num in range(total_batches):
        start_time = time.time()

        # Determine the start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(pmcids_list))

        # Process the PMCIDs in this batch
        for i in range(start_idx, end_idx):
            pmcid = pmcids_list[i]
            #print(f"Processing {pmcid}...")

            n+=1 
            
            
            numeric_pmcid = pmcid.replace('PMC', '', 1)  # Remove 'PMC' prefix if it appears at the beginning
            handle = Entrez.elink(dbfrom="pmc", db="pubmed", id=numeric_pmcid, retmode="xml")
            response = handle.read()
            handle.close()
            root = ET.fromstring(response)

            pmid = None
            for linksetdb in root.findall(".//LinkSetDb"):
                if linksetdb.findtext("DbTo") == 'pubmed':
                    pmid = linksetdb.findtext(".//Id")
                    break

            if pmid:
                #print(f"Found PMID {pmid} for {pmcid}.")
                pmid_dict[pmcid] = pmid
            else:
                print(f"No PMID found for {pmcid}.\n")

        # Sleep for a short period to avoid overwhelming the service
        time.sleep(3)
        print(n, ' PMCIDs processed out of ', len(pmcids_list))

        # Calculate and print the time taken for this batch
        batch_time = time.time() - start_time
        print(f"Batch {batch_num + 1}/{total_batches} took {batch_time:.2f} seconds.")

        # Estimate the remaining time
        remaining_batches = total_batches - batch_num - 1
        remaining_time = batch_time * remaining_batches
        print(f"Estimated remaining time: {remaining_time / 60:.2f} minutes.")

    return pmid_dict

   
####################

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, required=True, help='Path to the working directory where all files are')

# Inputs: 
parser.add_argument('--biomes_df', type=str, required=True, help='this is the sample file with biomes')

parser.add_argument('--pmids_dict_path', type=str, required=True, help='pmids scraped from metadata')
parser.add_argument('--pmcids_dict_path', type=str, required=True, help='pmcids scraped from metadata')

parser.add_argument('--dois_pmids_dict_path', type=str, required=True, help='pmids obtained from dois (scraped from metadata)')

parser.add_argument('--bioprojects_pmcid_dict_path', type=str, required=True, help='pmcids obtained from bioprojects (scraped from metadata)')

# Outputs
parser.add_argument('--output_file', type=str, required=True, help='name of output file')
parser.add_argument('--figure', type=str, required=True, help='name of output plot')

args = parser.parse_args()

# Prepend work_dir to all the file paths
biomes_df = os.path.join(args.work_dir, args.biomes_df)
pmids_dict_path = os.path.join(args.work_dir, args.pmids_dict_path)
pmcids_dict_path = os.path.join(args.work_dir, args.pmcids_dict_path)
dois_pmids_dict_path = os.path.join(args.work_dir, args.dois_pmids_dict_path)
bioprojects_pmcid_dict_path = os.path.join(args.work_dir, args.bioprojects_pmcid_dict_path)
output_file = os.path.join(args.work_dir, args.output_file)
figure = os.path.join(args.work_dir, args.figure)


####################
# for testing purposes 
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/'

biomes_df = work_dir + 'samples_biomes' 

pmids_dict_path = work_dir + 'sample.info_pmid' 
pmcids_dict_path = work_dir + 'sample.info_pmcid' 

dois_pmids_dict_path = work_dir + 'sample.info_doi' 

bioprojects_pmcid_dict_path = work_dir + 'sample.info_bioproject' 

output_file = work_dir + "sample.info_biome_pmid.csv"
figure = work_dir + "sample.info_biome_pmid.pdf"
####################



##################### Part 1. open dictionary files: 

samples_biomes = read_json_file(biomes_df)
    
a = read_json_file(pmids_dict_path)
b = read_json_file(dois_pmids_dict_path)


c = read_json_file(pmcids_dict_path)
d = read_json_file(bioprojects_pmcid_dict_path)
    

##################### Part 2. merge dictionaries

merged_a_b = merge_dicts(a, b)
len(merged_a_b)
merged_c_d = merge_dicts(c, d)
len(merged_c_d)


#merged_c_d = {k: merged_c_d[k] for k in list(merged_c_d)[:5000]}

##################### Part 3. 

unique_a_b = unique_values(merged_a_b)
len(unique_a_b) 
# 250
unique_c_d = unique_values(merged_c_d)
len(unique_c_d) 
# 4997 


#################### Part 4. transform unique pmcids to pmids and replace in dictionary:

z2 = from_pmcids_to_pmids(unique_c_d) # test with: unique_c_d[1:100]
z2


new_dict = {}
for key, values in merged_c_d.items():
    new_values = [z2.get(value, value) for value in values]
    new_dict[key] = new_values

# check if all PMC values have been replaced:
count = sum(1 for values in new_dict.values() for item in values if item.startswith('PMC'))
print(count, ' PMCIDs have not been translated to PMIDs')
# 237 



#################### Part 5. Merge all into one dataframe 

# 1. samples_biomes # careful opening it, it crashes
def head(dictionary, n=5):
    return dict(islice(dictionary.items(), n))
print(head(samples_biomes, 10))  # Prints the first 10 items
# Convert to df
df_samples = pd.DataFrame(list(samples_biomes.items()), columns=['sample', 'biome'])


# 2. all samples-pmids:
merged_a_b_c_d = merge_dicts(merged_a_b, new_dict)
print(head(merged_a_b_c_d, 5))  # Prints the first 10 items
# Convert to df
merged_a_b_c_d_df = pd.DataFrame(list(merged_a_b_c_d.items()), columns=['sample', 'pmid'])



# 3. merge all: 
merged_all = pd.merge(df_samples, merged_a_b_c_d_df, on='sample', how='outer')
print(merged_all)

# Save:
merged_all.to_csv(output_file, index=False)



#################### Part 6. Plot the content


# =============================================================================
# # Sample dataframe
# data = {
#     'sample': ['1', '2', '3', '4','5'],
#     'biome': ['animal', 'animal', 'water', 'water','soil'],
#     'pmid': ['a,b','','','a','c'],
#     'pmcid': ['x,z,y','y','','v,w','']
# }
# 
# df = pd.DataFrame(data)
# =============================================================================

df = merged_all

# Calculate counts
total_samples_per_biome = df.groupby('biome').size()
pmid_per_biome = df[df['pmid'].isna()].groupby('biome').size()

def unique_pmids(pmid_lists):
    # Filter out None or NaN values
    valid_lists = pmid_lists.dropna()
    
    # Flatten the list of lists
    flattened = [pmid for sublist in valid_lists for pmid in sublist]
    
    return len(set(flattened))

unique_pmids_per_biome = df.groupby('biome')['pmid'].agg(unique_pmids)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.3
positions = np.arange(len(total_samples_per_biome))

total_samples_per_biome.plot(kind='bar', position=0, width=bar_width, ax=ax, label='Total Samples', color='blue')
pmid_per_biome.plot(kind='bar', position=1, width=bar_width, ax=ax, label='Samples with PMID', color='red')
unique_pmids_per_biome.plot(kind='bar', position=2, width=bar_width, ax=ax, label='Unique PMIDs', color='green')

# Add annotations
for idx, rect in enumerate(ax.patches):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 5, '%d' % int(height), ha='center', va='bottom', rotation=90)

ax.set_ylabel('Count')
ax.set_title('Biome-wise Analysis')
ax.legend()

plt.xticks(positions + bar_width, total_samples_per_biome.index, rotation=0)
plt.tight_layout()


# Save the plot
plt.savefig(figure, dpi=300, bbox_inches='tight')

plt.show()

elapsed_time = time.time() - start_time
print(f"Code ran in {elapsed_time:.2f} seconds ")





