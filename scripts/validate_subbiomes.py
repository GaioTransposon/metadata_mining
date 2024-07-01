#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:17:22 2024

@author: dgaio
"""

import sys 
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from embeddings_functions import load_embeddings
from embeddings_functions import compare_embeddings
from features_process import filter_common_keys
from embeddings_functions import print_statistics
from embeddings_functions import plot_distribution_metrics
from embeddings_functions import plot_comparison_distribution
from embeddings_functions import create_shuffled_background_distribution
from embeddings_functions import sample_by_category
from embeddings_functions import plot_heatmap
from embeddings_functions import save_figures_to_pdf
import os
import numpy as np
import random
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel


# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'

###
# Gold dict path 
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')  
embeddings_gd = load_embeddings(gold_dict_json_path)
###

########
# expected different
my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']


# expected similar
# my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
#             'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs22_API118_normal_dt202406051326.txt']


gpt_json_files=[]
for i in my_files:
   j=i.replace('.txt', '_sbembeddings.json')
   gpt_json_files.append(j)
########




# Fetch embeddings from each gpt json file and compare to ground truth embeddings:
for gpt_file in gpt_json_files:
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    embeddings_gpt = load_embeddings(gpt_json_path)
    
    # Filter embeddings to include only common keys
    filtered_gd, filtered_gpt = filter_common_keys(embeddings_gd, embeddings_gpt)
    print("Filtered gold_dict_sb:", len(filtered_gd))
    print("Filtered gpt_clean_sb:", len(filtered_gpt))

    ########################################
    # Compare embeddings
    compare_results = compare_embeddings(filtered_gd, filtered_gpt)

    ########################################
    # Calculate and print statistics
    actual_similarities = [result['cosine'] for result in compare_results.values()]
    avg_sim, median_sim = print_statistics(actual_similarities)

    ########################################
    # Plot distribution metrics
    plot_distribution_metrics(compare_results)

    ########################################
    # Plot similarity vs background
    background_similarities = create_shuffled_background_distribution(filtered_gd, filtered_gpt, num_comparisons=len(actual_similarities))
    comparison_fig = plot_comparison_distribution(actual_similarities, background_similarities, avg_sim, median_sim)

    ########################################
    # Extract sub_biome_texts directly from embeddings dictionaries
    gold_labels = {key: embeddings_gd[key]['sub-biome'] for key in embeddings_gd}
    gpt_labels = {key: embeddings_gpt[key]['sub-biome'] for key in embeddings_gpt}
    
    ### # filtering to keep only common keys in both dictionaries, and 10 per biome
    gold_biomes = {key: embeddings_gd[key]['biome'] for key in embeddings_gd}
    common_keys = list(gold_labels.keys() & gpt_labels.keys())
    sampled_keys = sample_by_category(common_keys, gold_biomes, 10)
    ###
    
    
    # Prepare data matrices for the heatmap
    matrix_gd = np.array([embeddings_gd[key]['embedding'] for key in sampled_keys])
    matrix_gpt = np.array([embeddings_gpt[key]['embedding'] for key in sampled_keys])

    # Extract the labels for the sampled keys
    gold_labels_sampled = {key: gold_labels[key] for key in sampled_keys}
    gpt_labels_sampled = {key: gpt_labels[key] for key in sampled_keys}

    # Generate heatmap
    heatmap_fig = plot_heatmap(matrix_gd, matrix_gpt, gpt_labels_sampled, gold_labels_sampled, sampled_keys, sampled_keys)


    ########################################
    # Save both figures to a PDF
    gpt_base_file = gpt_file.replace('_sbembeddings.json', '')
    save_figures_to_pdf([comparison_fig, heatmap_fig], gpt_base_file, embeddings_dir)
    
    
    
    
    
    
    
    
    




import json
import os

# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'

# expected different
my_files = ['gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
            'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt']



# Gold dictionary file
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')

# Function to load embeddings and extract sub-biome assignments
def extract_sub_biomes(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    sub_biomes = {k: v['sub-biome'] for k, v in data.items()}
    return sub_biomes

# Extract sub-biome assignments for GPT files
gpt_sub_biomes = {}
for gpt_file in gpt_json_files:
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    gpt_sub_biomes[gpt_file] = extract_sub_biomes(gpt_json_path)

# Extract sub-biome assignments for gold dictionary
gold_sub_biomes = extract_sub_biomes(gold_dict_json_path)

# Print header
print("Sample ID\tGold Dict Sub-Biome\tGPT File 1 Sub-Biome\tGPT File 2 Sub-Biome")

# Extract sample IDs common to all
common_sample_ids = set(gold_sub_biomes.keys())
for gpt_sub_biome in gpt_sub_biomes.values():
    common_sample_ids &= set(gpt_sub_biome.keys())

# Print sub-biomes for each common sample ID
for sample_id in common_sample_ids:
    gold_sub_biome = gold_sub_biomes[sample_id]
    gpt_sub_biome1 = gpt_sub_biomes[gpt_json_files[0]].get(sample_id, "N/A")
    gpt_sub_biome2 = gpt_sub_biomes[gpt_json_files[1]].get(sample_id, "N/A")
    print(f"{sample_id}\t#{gold_sub_biome}\t#{gpt_sub_biome1}\t#{gpt_sub_biome2}")












