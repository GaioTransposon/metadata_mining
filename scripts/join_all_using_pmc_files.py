#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:45:40 2023

@author: dgaio
"""

# # Before running: 
# # from Terminal within MicrobeAtlasProject/: 
# wget https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz
# wget https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_comm_use_file_list.csv
# wget https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_non_comm_use_pdf.csv  
# gzip -d PMC-ids.csv.gz  

# # run as: 
# python ~/github/metadata_mining/scripts/join_all_using_pmc_files.py  \
#         --work_dir '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/' \
#             --biomes_df 'samples_biomes' \
#                 --pmcids_to_pmids1 'PMC-ids.csv' \
#                     --pmcids_to_pmids2 'oa_comm_use_file_list.csv' \
#                         --pmcids_to_pmids3 'oa_non_comm_use_pdf.csv' \
#                             --pmids_dict_path 'sample.info_pmid' \
#                                 --pmcids_dict_path 'sample.info_pmcid' \
#                                     --dois_pmids_dict_path 'sample.info_doi' \
#                                         --bioprojects_pmcid_dict_path 'sample.info_bioproject' \
#                                             --output_file 'sample.info_biome_pmid.csv' \
#                                                 --figure 'sample.info_biome_pmid.pdf' 

import os
import time
import argparse
import json
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


def extract_pmcid_pmid_using_files(*files):
    dfs = []

    # Mapping of possible column names to standard names
    column_mappings = {'PMCID': 'PMCID', 'Accession ID': 'PMCID', 'PMID': 'PMID'}
    
    for file in files:
        # Check which columns are available in the current file
        available_cols = pd.read_csv(file, nrows=0).columns.intersection(column_mappings.keys()).tolist()
        
        if not available_cols:
            continue  # If neither 'PMCID', 'Accession ID' nor 'PMID' are available, skip

        # Read only the relevant columns using pandas
        data = pd.read_csv(file, usecols=available_cols, dtype=str)

        # Rename columns based on our mappings
        data = data.rename(columns=column_mappings)

        # Drop rows where any of the columns have NaN values
        data = data.dropna(subset=['PMCID', 'PMID'])

        dfs.append(data)

    # Concatenate all DataFrames
    df = pd.concat(dfs, ignore_index=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def translate_pmcid_to_pmid(pmcid_list, result_df):
    # Filter the dataframe based on the pmcid_list
    filtered_df = result_df[result_df['PMCID'].isin(pmcid_list)]
    
    # Convert the filtered dataframe to a dictionary
    translation_dict = dict(zip(filtered_df['PMCID'], filtered_df['PMID']))
    
    return translation_dict

   
####################

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, required=True, help='Path to the working directory where all files are')

# Inputs: 
parser.add_argument('--biomes_df', type=str, required=True, help='this is the sample file with biomes')
parser.add_argument('--pmcids_to_pmids1', type=str, required=True, help='pmcids_to_pmids1')
parser.add_argument('--pmcids_to_pmids2', type=str, required=True, help='pmcids_to_pmids2')
parser.add_argument('--pmcids_to_pmids3', type=str, required=True, help='pmcids_to_pmids3')

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
pmcids_to_pmids1 = os.path.join(args.work_dir, args.pmcids_to_pmids1)
pmcids_to_pmids2 = os.path.join(args.work_dir, args.pmcids_to_pmids2)
pmcids_to_pmids3 = os.path.join(args.work_dir, args.pmcids_to_pmids3)
pmids_dict_path = os.path.join(args.work_dir, args.pmids_dict_path)
pmcids_dict_path = os.path.join(args.work_dir, args.pmcids_dict_path)
dois_pmids_dict_path = os.path.join(args.work_dir, args.dois_pmids_dict_path)
bioprojects_pmcid_dict_path = os.path.join(args.work_dir, args.bioprojects_pmcid_dict_path)
output_file = os.path.join(args.work_dir, args.output_file)
figure = os.path.join(args.work_dir, args.figure)


# =============================================================================
# ####################
# # for testing purposes 
# work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/'
# 
# biomes_df = work_dir + 'samples_biomes' 
# pmcids_to_pmids1 = work_dir + 'PMC-ids.csv' 
# pmcids_to_pmids2 = work_dir + 'oa_comm_use_file_list.csv' 
# pmcids_to_pmids3 = work_dir + 'oa_non_comm_use_pdf.csv' 
# 
# pmids_dict_path = work_dir + 'sample.info_pmid' 
# pmcids_dict_path = work_dir + 'sample.info_pmcid' 
# 
# dois_pmids_dict_path = work_dir + 'sample.info_doi' 
# 
# bioprojects_pmcid_dict_path = work_dir + 'sample.info_bioproject' 
# 
# output_file = work_dir + "sample.info_biome_pmid.csv"
# figure = work_dir + "sample.info_biome_pmid.pdf"
# ####################
# =============================================================================



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

# # for testing purposes
# merged_c_d = {k: merged_c_d[k] for k in list(merged_c_d)[:5000]}

##################### Part 3. 

unique_a_b = unique_values(merged_a_b)
len(unique_a_b) 
# 250
unique_c_d = unique_values(merged_c_d)
len(unique_c_d) 
# 4997 


#################### Part 4. transform unique pmcids to pmids and replace in dictionary:


files = [pmcids_to_pmids1, pmcids_to_pmids2, pmcids_to_pmids3]
pmcids_pmids_files = extract_pmcid_pmid_using_files(*files)
print(len(pmcids_pmids_files))
print(pmcids_pmids_files.tail(3))


translations = translate_pmcid_to_pmid(unique_c_d, pmcids_pmids_files)
print(dict(islice(translations.items(), 10)))
print(len(translations))


z2 = translations


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


