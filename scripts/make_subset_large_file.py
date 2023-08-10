#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:58:43 2023

@author: dgaio
"""


# # run as: 
# python ~/github/metadata_mining/scripts/make_subset_large_file.py  \
#     --large_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info'  \
#         --biomes_df "~/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv"  \
#             --output_samples_biome_dict "~/cloudstor/Gaio/MicrobeAtlasProject/samples_biomes" \
#                 --output_subset "~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_subset" 



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import argparse  
import os 
import json



def save_to_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)
  


def get_keys_from_dict(dictionary):
    if isinstance(dictionary, str):
        # If dictionary is a string, attempt to load it as JSON
        import json
        dictionary = json.loads(dictionary)

    return set(dictionary.keys())



def subset_large_file(input_file_path, output_file_path, valid_samples_set):
    with open(input_file_path, 'r') as input_file:
        with open(output_file_path, 'w') as output_file:
            writing_sample = False
            for line in input_file:
                if line.startswith('>'):
                    sample_name = line.replace('>', '').strip()
                    writing_sample = sample_name in valid_samples_set

                if writing_sample:
                    output_file.write(line)



parser = argparse.ArgumentParser(description='make sample.info subset based on samples in Janko s file')
parser.add_argument('--large_file', type=str, required=True, help='Path to sample.info')
parser.add_argument('--biomes_df', type=str, required=True, help='Path to Janko''s file')
parser.add_argument('--output_samples_biome_dict', type=str, required=True, help='name of output for dictionary sample-biome')
parser.add_argument('--output_subset', type=str, required=True, help='name of output for sample.info subset')
args = parser.parse_args()

large_file = os.path.expanduser(args.large_file)
biomes_df = os.path.expanduser(args.biomes_df)
output_samples_biome_dict = os.path.expanduser(args.output_samples_biome_dict)
output_subset = os.path.expanduser(args.output_subset)

# # for testing purposes 
# large_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info' 
# biomes_df = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv' 
# output_samples_biome_dict = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/samples_biomes' 
# output_subset = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_subset' 



# open and transform also to dictionary: 
biomes_df = pd.read_csv(biomes_df, sep='\t')
# some df parse and col rename
biomes_df['sample'] = biomes_df['SampleID'].str.split('.').str[1]
sample_biome_dict = dict(zip(biomes_df['sample'], biomes_df['EnvClean_merged']))



save_to_json(sample_biome_dict, output_samples_biome_dict)

biomes_dict_to_set = get_keys_from_dict(sample_biome_dict)

subset_large_file(large_file, output_subset, biomes_dict_to_set)








