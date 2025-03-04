#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:28:44 2024

@author: dgaio
"""

import json
import numpy as np
import random
from scipy.spatial import distance
from matplotlib.backends.backend_pdf import PdfPages
import random
from scipy.stats import ttest_rel, ttest_ind
#from statsmodels.stats.multitest import multipletests


# Now adapted to load embeddings with this format: {sample_id: {'embedding': [values], 'sub_biome_text': text}}
def load_embeddings(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Create a dictionary to store embeddings, sub_biome_text, and biome (if available)
    embeddings_dict = {}
    for k, v in data.items():
        embeddings = np.array(v['embedding'], dtype=np.float32)
        sub_biome_text = v['sub-biome']
        embeddings_dict[k] = {'embedding': embeddings, 'sub-biome': sub_biome_text}
        
        # Include biome if it exists in the data
        if 'biome' in v:
            embeddings_dict[k]['biome'] = v['biome']

    return embeddings_dict


def compare_embeddings(embeddings_dict1, embeddings_dict2):
    comparison_results = {}
    for sample_id, data1 in embeddings_dict1.items():
        if sample_id in embeddings_dict2:
            # Extract embedding arrays from each dictionary
            embedding1 = data1['embedding']
            embedding2 = embeddings_dict2[sample_id]['embedding']
            
            # Compute distances and similarity
            euclidean_dist = distance.euclidean(embedding1, embedding2)
            cosine_sim = 1 - distance.cosine(embedding1, embedding2)  # cosine similarity is 1 - cosine distance
            manhattan_dist = distance.cityblock(embedding1, embedding2)
            
            # Store results
            comparison_results[sample_id] = {
                'euclidean': euclidean_dist,
                'cosine': cosine_sim, 
                'manhattan': manhattan_dist
            }
    return comparison_results


def create_shuffled_background_distribution(embeddings_gd, embeddings_gpt, num_comparisons=None):
    random_cosine_similarities = []
    gd_keys = list(embeddings_gd.keys())
    gpt_keys = list(embeddings_gpt.keys())
    
    # Shuffle the keys for randomness in comparison
    random.shuffle(gpt_keys)
    
    # If num_comparisons is not specified, compare the full length of the smaller set
    if num_comparisons is None:
        num_comparisons = min(len(gd_keys), len(gpt_keys))
    else:
        num_comparisons = min(num_comparisons, len(gd_keys), len(gpt_keys))
    
    for i in range(num_comparisons):
        # Extract embeddings from each dictionary by specifying the 'embedding' key
        gd_embedding = embeddings_gd[gd_keys[i]]['embedding']
        gpt_embedding = embeddings_gpt[gpt_keys[i]]['embedding']
        
        # Calculate cosine similarity
        cosine_sim = 1 - distance.cosine(gd_embedding, gpt_embedding)
        random_cosine_similarities.append(cosine_sim)
    
    return random_cosine_similarities






def sample_by_category(common_keys, gold_biomes, n):
    random.seed(42)  # For reproducibility
    
    # Group common keys by biome
    category_groups = {}
    for key in common_keys:
        category = gold_biomes[key]  # Use the 'biome' value as the category
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(key)

    sampled_keys = []
    for category, keys in category_groups.items():
        if len(keys) > n:
            sampled_keys.extend(random.sample(keys, n))
        else:
            sampled_keys.extend(keys)  # If less than n, take all
    return sampled_keys





        
        