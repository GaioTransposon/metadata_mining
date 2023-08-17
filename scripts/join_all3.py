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
#                                 --output_file 'sample.info_biome_pmid_pmcid' \
#                                     --figure 'sample.info_biome_pmid_pmcid.pdf' \
#                                         --unique_pmids_pmcids 'unique_pmids_pmcids' 
## Code ran in 1268.51 seconds (~20min)

import os
import time
import argparse
from Bio import Entrez
import json
import xml.etree.ElementTree as ET
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt

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
    Entrez.email = "daniela.gaio@mls.uzh.ch"
    pmid_dict = {}
    
    n=0
    for pmcid in pmcids:
        numeric_pmcid = pmcid.replace('PMC', '', 1)  # Remove the 'PMC' prefix only if it appears at the beginning
        handle = Entrez.elink(dbfrom="pmc", db="pubmed", id=numeric_pmcid, retmode="xml")
        response = handle.read()
        handle.close()
        root = ET.fromstring(response)
        
        pmid = None
        for linksetdb in root.findall(".//LinkSetDb"):
            if linksetdb.findtext("DbTo") == 'pubmed':
                pmid = linksetdb.findtext(".//Id")
                break
        
        if pmid: # only store if pmcid is not None
            pmid_dict[pmcid] = pmid
        
        n += 1
        print(n, 'pmcids handled out of ', len(pmcids))
    
    
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
parser.add_argument('--unique_pmids_pmcids', type=str, required=True, help='name of file of unique pmids and unique pmcids as output')

args = parser.parse_args()

# Prepend work_dir to all the file paths
biomes_df = os.path.join(args.work_dir, args.biomes_df)
pmids_dict_path = os.path.join(args.work_dir, args.pmids_dict_path)
pmcids_dict_path = os.path.join(args.work_dir, args.pmcids_dict_path)
dois_pmids_dict_path = os.path.join(args.work_dir, args.dois_pmids_dict_path)
bioprojects_pmcid_dict_path = os.path.join(args.work_dir, args.bioprojects_pmcid_dict_path)
output_file = os.path.join(args.work_dir, args.output_file)
figure = os.path.join(args.work_dir, args.figure)
unique_pmids_pmcids = os.path.join(args.work_dir, args.unique_pmids_pmcids)


####################
# =============================================================================
# # for testing purposes 
# work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/'
# 
# biomes_df = work_dir + 'samples_biomes' 
# 
# pmids_dict_path = work_dir + 'sample.info_pmid' 
# pmcids_dict_path = work_dir + 'sample.info_pmcid' 
# 
# dois_pmids_dict_path = work_dir + 'sample.info_doi' 
# 
# bioprojects_pmcid_dict_path = work_dir + 'sample.info_bioproject' 
# 
# output_file = work_dir + "sample.info_biome_pmid_pmcid"
# figure = work_dir + "sample.info_biome_pmid_pmcid.pdf"
# unique_pmids_pmcids = work_dir + "unique_pmids_pmcids"
# =============================================================================
####################
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


##################### Part 3. 

unique_a_b = unique_values(merged_a_b)
len(unique_a_b) 
# 250
unique_c_d = unique_values(merged_c_d)
len(unique_c_d) 
# 1575


#################### Part 4. transform unique pmcids to pmids and replace in dictionary:

z2 = from_pmcids_to_pmids(unique_c_d) # test with: unique_c_d[1:100]
z2


new_dict = {}
for key, values in merged_c_d.items():
    new_values = [z2.get(value, value) for value in values]
    new_dict[key] = new_values
print(new_dict)


# check if all PMC values have been replaced:
count = sum(1 for values in new_dict.values() for item in values if item.startswith('PMC'))
print('If zero, all PMCIDs have been translated to PMIDs: ', count)




#################### Part 5. Point of the situation: these 4 files to go on with: 


# 1. samples_biomes # careful opening it, it crashes
def head(dictionary, n=5):
    return dict(islice(dictionary.items(), n))
print(head(samples_biomes, 10))  # Prints the first 10 items

# 2. all samples-pmids:
merged_a_b_c_d = merge_dicts(merged_a_b, new_dict)
print(head(merged_a_b_c_d, 10))  # Prints the first 10 items










#################### Part 6. Merge all into one dataframe and plot the content


# Convert dictionaries to DataFrames
df_samples = pd.DataFrame(list(samples_biomes.items()), columns=['sample', 'biome'])
df_merged = pd.DataFrame(list(merged_a_b_c_d.items()), columns=['sample', 'pmid'])

# Merge the two DataFrames on the 'sample' column
merged_df = pd.merge(df_samples, df_merged, on='sample', how='outer')

print(merged_df)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

df = merged_df

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
plt.show()













# only for testing: 
# samples_biomes = {'sample1': 'animal', 'sample2': 'animal', 'sample3': 'soil'}
# merged_a_b = {'sample1': ['123','234'], 'sample2': ['789']}  # shortened for brevity
# merged_c_d = {'sample2': ['PMC567'], 'sample3': ['PMC00']}  # shortened for brevity
# z2 = {'PMC567': 'new', 'PMCnew2': '123'}

# Convert to dataframes
df_biomes = pd.DataFrame(list(samples_biomes.items()), columns=['sample', 'biome'])
df_pmids = pd.DataFrame(list(merged_a_b.items()), columns=['sample', 'pmid'])
df_pmcids = pd.DataFrame(list(merged_c_d.items()), columns=['sample', 'pmcid'])

# Merge dataframes
merged_df = df_biomes.merge(df_pmids, on='sample', how='left').merge(df_pmcids, on='sample', how='left')

def update_pmids(row):
    pmids = row['pmid'] if isinstance(row['pmid'], list) else []
    pmcids_translated = [z2.get(pmc, None) for pmc in row['pmcid']] if isinstance(row['pmcid'], list) else []
    pmcids_translated = [x for x in pmcids_translated if x is not None]
    return list(set(pmids + pmcids_translated))

def update_pmcids(row):
    pmids = row['pmid'] if isinstance(row['pmid'], list) else []
    pmcids = row['pmcid'] if isinstance(row['pmcid'], list) else []
    
    # Check if any pmid corresponds to a key in z2
    for pmid in pmids:
        matching_pmcid = [k for k, v in z2.items() if v == pmid]
        pmcids.extend(matching_pmcid)
    
    return list(set(pmcids))

merged_df['pmid'] = merged_df.apply(update_pmids, axis=1)
merged_df['pmcid'] = merged_df.apply(update_pmcids, axis=1)

# Convert lists to comma-separated strings
merged_df['pmid'] = merged_df['pmid'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
merged_df['pmcid'] = merged_df['pmcid'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)

len(merged_df)
print(merged_df)


# Save the original DataFrame (before editing)
merged_df.to_csv(output_file, index=False)


#################### Part 7. Plot the content of the dataframe


# Extract unique PMIDs and PMCIDs
all_pmids = set(filter(None, ','.join(merged_df['pmid']).split(',')))
len(all_pmids)
all_pmcids = set(filter(None, ','.join(merged_df['pmcid']).split(',')))
len(all_pmcids)


# Combine the unique PMIDs and PMCIDs
combined_ids = all_pmids.union(all_pmcids)
len(combined_ids)

# Save the combined set to a file
with open(unique_pmids_pmcids, 'w') as f:
    for unique_id in combined_ids:
        f.write(unique_id + '\n')












# for testing purposes: 
# =============================================================================
# # Sample dataframe
# data = {
#     'sample': ['1', '2', '3', '4','5'],
#     'biome': ['animal', 'animal', 'water', 'water','soil'],
#     'pmid': ['a,b','','','a','c'],
#     'pmcid': ['x,z,y','y','','v,w','']
# }
# 
# merged_df = pd.DataFrame(data)
# =============================================================================

# Functions to gather unique PMIDs and PMCIDs
def gather_unique_ids(x):
    return set(filter(None, ','.join(x).split(',')))

# Create columns indicating presence of non-NaN data
merged_df['pmid_present'] = merged_df['pmid'].apply(lambda x: 0 if x == '' else 1)
merged_df['pmcid_present'] = merged_df['pmcid'].apply(lambda x: 0 if x == '' else 1)

# Gather unique PMIDs and PMCIDs per biome
biome_unique_pmids = merged_df.groupby('biome')['pmid'].apply(gather_unique_ids).reset_index()
biome_unique_pmids['unique_pmids'] = biome_unique_pmids['pmid'].apply(len)
biome_unique_pmcids = merged_df.groupby('biome')['pmcid'].apply(gather_unique_ids).reset_index()
biome_unique_pmcids['unique_pmcids'] = biome_unique_pmcids['pmcid'].apply(len)

# Group by biome and sum/aggregate the created columns
biome_counts = merged_df.groupby('biome').agg({
    'pmid_present': 'sum',
    'pmcid_present': 'sum',
    'sample': 'size'
}).reset_index()

# Merge the counts with the unique counts
biome_counts = biome_counts.merge(biome_unique_pmids[['biome', 'unique_pmids']], on='biome').merge(biome_unique_pmcids[['biome', 'unique_pmcids']], on='biome')


# Plot
ax = biome_counts.set_index('biome').plot(kind='bar', figsize=(12,8))

# Add counts on top of the bars with vertical rotation
for p in ax.patches:
    ax.annotate(str(p.get_height()), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), textcoords='offset points', 
                rotation=90)

plt.title('Count of Rows per Biome')
plt.ylabel('Number of Rows/Unique IDs')
plt.xlabel('Biome')
plt.xticks(rotation=45)
plt.legend(['PMID Present', 'PMCID Present', 'Total Rows', 'Unique PMIDs', 'Unique PMCIDs'])
plt.tight_layout()

# Save the plot
plt.savefig(figure, dpi=300, bbox_inches='tight')

plt.show()



elapsed_time = time.time() - start_time
print(f"Code ran in {elapsed_time:.2f} seconds ")



