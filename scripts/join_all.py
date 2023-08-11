#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 23:24:40 2023

@author: dgaio
"""


import os
import re
import time
import argparse
from Bio import Entrez
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
from itertools import islice



# for testing purposes 
biomes_df = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/samples_biomes' 

pmids_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid' 
pmcids_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmcid' 

dois_pmids_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_doi' 

bioprojects_pmcid_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_bioproject' 

output_file = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid_pmcid_biome.csv"





def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    


samples_biomes = read_json_file(biomes_df)
    
a = read_json_file(pmids_dict_path)
b = read_json_file(dois_pmids_dict_path)


c = read_json_file(pmcids_dict_path)
d = read_json_file(bioprojects_pmcid_dict_path)
    



def merge_dicts(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            result[key] = list(set(result[key] + value))
        else:
            result[key] = value
    return result



merged_a_b = merge_dicts(a, b)
len(merged_a_b)
merged_c_d = merge_dicts(c, d)
len(merged_c_d)




def unique_values(dictionary):
    unique_vals = set()
    for values in dictionary.values():
        if isinstance(values, list):
            unique_vals.update(values)
        else:
            unique_vals.add(values)
    return list(unique_vals)




unique_c_d = unique_values(merged_c_d)
len(unique_c_d) 
# 1575








def from_pmcids_to_pmids(pmcids):
    Entrez.email = "your.email@example.com"
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





# pmids = ["34131077", "23456789","34131077", "000349299200040"]
# pmcids = ["PMC8237688", "8237688", "PMC6121709"]  


z1 = from_pmids_to_pmcids(unique_a_b[1:10])
z1


from_pmids_to_pmcids(['18769457'])



z2 = from_pmcids_to_pmids(unique_c_d[11:20])
z2






# go on with :
    

# 1. samples_biomes # careful opening it, crashes
def head(dictionary, n=5):
    return dict(islice(dictionary.items(), n))

print(head(samples_biomes, 10))  # Prints the first 10 items

# 2. all pmids:
merged_a_b

# 3. all pmids:
merged_c_d
    
# 3. transaltions of unique pmcids to pmids: 
z2






    

# Function to combine information from multiple dictionaries
def combine_info(combined_dict, *sample_infos):
    for sample_info in sample_infos:
        for sample, values in sample_info.items():
            for value in values:
                key = 'PMCIDs' if value.startswith('PMC') else 'PMIDs'
                # No need to check for sample existence since samples_biomes contains all the samples
                combined_dict[sample][key].append(value)
    return combined_dict


def get_unique_ids(sample_dict):
    unique_pmids = set()
    unique_pmcids = set()
    
    for sample_info in sample_dict.values():
        unique_pmids.update(sample_info['PMIDs'])
        unique_pmcids.update(sample_info['PMCIDs'])
    
    return list(unique_pmids), list(unique_pmcids)



unique_pmids, unique_pmcids = get_unique_ids(z)
print("Unique PMIDs:", unique_pmids)
print("Unique PMCIDs:", unique_pmcids)


    















combined_dict2 = combined_dict



# Update combined_dict based on the mappings
for key, value in combined_dict2.items():
    for pmid in value['PMIDs']:
        if pmid in pmids_to_pmcids and pmids_to_pmcids[pmid] not in value['PMCIDs']:
            value['PMCIDs'].append(pmids_to_pmcids[pmid])
    
    for pmcid in value['PMCIDs']:
        if pmcid in pmcids_to_pmids and pmcids_to_pmids[pmcid] not in value['PMIDs']:
            value['PMIDs'].append(pmcids_to_pmids[pmcid])