#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:39:52 2024

@author: dgaio
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import os
from collections import Counter
import textwrap



GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"


def load_gold_dict(path):
    with open(path, 'rb') as file:
        data, processed_pmids = pickle.load(file)
    return data
    


def analyze_gold_dict(gold_dict):
    biome_counter = Counter()
    sub_biome_counter = {}
    # count biomes and sub-biomes
    for entry in gold_dict.values():        
        if len(entry)>2: 
            biome = entry[1]
            #print(biome)
            sub_biome = entry[2]
            #print(sub_biome)
            biome_counter[biome] += 1
            if biome not in sub_biome_counter:
                sub_biome_counter[biome] = Counter()
            sub_biome_counter[biome][sub_biome] += 1

    return biome_counter, sub_biome_counter



def create_bar_plots(sub_biome_counter):
    for biome, counter in sub_biome_counter.items():
        filtered_counter = {k: v for k, v in counter.items() if v > 1}
        single_occurrence = {k: v for k, v in counter.items() if v == 1}
        
        sorted_sub_biomes = dict(sorted(filtered_counter.items(), key=lambda item: item[1], reverse=True))

        num_sub_biomes = len(sorted_sub_biomes)
        fig_width = 14
        fig_height = max(6, 0.5 * num_sub_biomes)  # 6 inches tall

        plt.figure(figsize=(fig_width, fig_height))
        keys = list(sorted_sub_biomes.keys())
        values = list(sorted_sub_biomes.values())

        plt.barh(keys, values)
        plt.xlabel('Frequency')
        plt.title(f'Sub-biome Frequency in {biome}')

        label_size = 14
        if num_sub_biomes > 20:
            label_size = 12
        elif num_sub_biomes > 10:
            label_size = 13
        plt.tick_params(axis='y', labelsize=label_size)

        plt.tight_layout()
        plt.show()

        if single_occurrence: 
            print(f"Sub-biomes in '{biome}' with a single occurrence:")
            for k, v in sorted(single_occurrence.items(), key=lambda item: item[0]):
                print(f"  {k}")




gold_dict = load_gold_dict(GOLD_DICT_PATH)

biome_counter, sub_biome_counter = analyze_gold_dict(gold_dict)
print(biome_counter)

create_bar_plots(sub_biome_counter)




# =============================================================================
# # Use to find sample IDs of unusual sub-biomes (to then edit with edi_gold_dict.py)
# def find_keys_by_sub_biome(gold_dict, sub_biome_str):
#     matching_keys = []
#     for key, value in gold_dict.items():
#         if len(value) >= 3 and sub_biome_str in value[2]:
#             matching_keys.append(key)
#     return matching_keys
# 
# sub_biome_to_search = 'NA'
# matching_keys = find_keys_by_sub_biome(gold_dict, sub_biome_to_search)
# print(f"Keys with sub-biome '{sub_biome_to_search}': {matching_keys}")
# print(len(matching_keys))
# =============================================================================




