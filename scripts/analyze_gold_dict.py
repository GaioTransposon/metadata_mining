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





# Define the path to the dictionary
GOLD_DICT_PATH = "/Users/dgaio/github/metadata_mining/source_data/gold_dict.pkl"

# Function to load the dictionary from a pickle file
def load_gold_dict(path):
    with open(path, 'rb') as file:
        data, processed_pmids = pickle.load(file)
    return data
    

# Function to analyze the dictionary
def analyze_gold_dict(gold_dict):
    biome_counter = Counter()
    sub_biome_counter = {}
    # Count biomes and sub-biomes
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



import matplotlib.pyplot as plt
from collections import Counter

def create_bar_plots(sub_biome_counter):
    for biome, counter in sub_biome_counter.items():
        plt.figure(figsize=(10, 5))
        keys = counter.keys()
        values = counter.values()
        
        # Creating bar plot
        plt.bar(keys, values)
        plt.title(f'Sub-biome Frequency in {biome}')
        plt.ylabel('Frequency')
        
        # Improve x-axis labels
        plt.xticks(rotation=90)  # Rotate labels to 90 degrees to prevent overlap
        plt.margins(x=0.01)  # Add some extra space on the x-axis
        plt.subplots_adjust(bottom=0.3)  # Adjust subplot to fit the x-axis labels
        
        # Automatic adjustment of label size depending on number of labels
        if len(keys) > 10:
            plt.tick_params(axis='x', labelsize=8)
        elif len(keys) > 5:
            plt.tick_params(axis='x', labelsize=10)
        
        # Show plot with tight layout
        plt.tight_layout()
        plt.show()
        

# Function to create horizontal bar plots for sub-biomes sorted by frequency
def create_bar_plots(sub_biome_counter):
    for biome, counter in sub_biome_counter.items():
        # Sort the counter by frequency
        sorted_sub_biomes = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))

        # Determine the number of sub-biomes to set figure size accordingly
        num_sub_biomes = len(sorted_sub_biomes)
        # Set the width of the figure to be proportional to the number of sub-biomes
        fig_width = 14
        fig_height = 6 * num_sub_biomes  # 0.5 inch per sub-biome

        plt.figure(figsize=(fig_width, fig_height))
        keys = list(sorted_sub_biomes.keys())
        values = list(sorted_sub_biomes.values())

        # Creating horizontal bar plot
        plt.barh(keys, values)
        plt.xlabel('Frequency')
        plt.title(f'Sub-biome Frequency in {biome}')

        # Adjust the size of the labels based on the number of sub-biomes
        label_size = 12
        if num_sub_biomes > 20:
            label_size = 6
        elif num_sub_biomes > 10:
            label_size = 7
        plt.tick_params(axis='y', labelsize=label_size)

        # Adjust layout to ensure everything fits
        plt.tight_layout()
        plt.show()

# Assume sub_biome_counter is already filled with data from previous steps.
# Call create_bar_plots with your data here.
# create_bar_plots(sub_biome_counter)

# Function to create horizontal bar plots for sub-biomes sorted by frequency,
# including only those sub-biomes with a frequency greater than one
def create_bar_plots(sub_biome_counter):
    for biome, counter in sub_biome_counter.items():
        # Filter out sub-biomes with a frequency of one
        filtered_counter = {k: v for k, v in counter.items() if v > 1}
        single_occurrence = {k: v for k, v in counter.items() if v == 1}
        
        # Sort the filtered counter by frequency
        sorted_sub_biomes = dict(sorted(filtered_counter.items(), key=lambda item: item[1], reverse=True))

        # Determine the number of sub-biomes to set figure size accordingly
        num_sub_biomes = len(sorted_sub_biomes)
        fig_width = 14
        fig_height = max(6, 0.5 * num_sub_biomes)  # At least 6 inches tall

        plt.figure(figsize=(fig_width, fig_height))
        keys = list(sorted_sub_biomes.keys())
        values = list(sorted_sub_biomes.values())

        # Creating horizontal bar plot
        plt.barh(keys, values)
        plt.xlabel('Frequency')
        plt.title(f'Sub-biome Frequency in {biome}')

        # Set the size of the labels
        label_size = 14
        if num_sub_biomes > 20:
            label_size = 12
        elif num_sub_biomes > 10:
            label_size = 13
        plt.tick_params(axis='y', labelsize=label_size)

        # Adjust layout to ensure everything fits
        plt.tight_layout()
        plt.show()

        # Print sub-biomes with a single occurrence to the console
        if single_occurrence: 
            print(f"Sub-biomes in '{biome}' with a single occurrence:")
            for k, v in sorted(single_occurrence.items(), key=lambda item: item[0]):
                print(f"  {k}")

# Example usage (assuming sub_biome_counter is already defined):
# create_bar_plots(sub_biome_counter)






# Load the dictionary
gold_dict = load_gold_dict(GOLD_DICT_PATH)

# Analyze the dictionary
biome_counter, sub_biome_counter = analyze_gold_dict(gold_dict)
print(biome_counter)

# Create bar plots
create_bar_plots(sub_biome_counter)





def find_keys_by_sub_biome(gold_dict, sub_biome_str):
    matching_keys = []
    for key, value in gold_dict.items():
        # Check if value has at least three elements and the third element contains the sub_biome_str
        if len(value) >= 3 and sub_biome_str in value[2]:
            matching_keys.append(key)
    return matching_keys



sub_biome_to_search = 'NA'
matching_keys = find_keys_by_sub_biome(gold_dict, sub_biome_to_search)
print(f"Keys with sub-biome '{sub_biome_to_search}': {matching_keys}")

len(matching_keys)




