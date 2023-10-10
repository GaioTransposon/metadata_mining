#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:05:41 2023

@author: dgaio
"""

import glob
import pandas as pd
import os 
import openai
import pandas as pd
import numpy as np  
from collections import Counter
import argparse  
import pickle
import re
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




# for testing purposes
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
input_df = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")


# 1. Load all files
file_pattern = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gtp_35_output*"
all_files = glob.glob(file_pattern)


# Your gold standard df, for example purposes
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")

########### # 2. open gold_dict and transform to a df
with open(input_gold_dict, 'rb') as file:
    input_gold_dict = pickle.load(file)

input_gold_dict = input_gold_dict[0] # this is because the second item is the list of pmids I processed - it was necessary to run confirm_biome_game.py, now not necessary anymore 
gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)




# Assuming gold_standard_df exists with columns 'sample' and 'curated_biome'
# Assuming all_files list has been populated

def load_and_process_file(file_name, gold_standard_df):
    dfr = pd.read_csv(file_name, sep=": ", engine='python', header=None, names=["sample", "gpt_generated_biome"])
    return pd.merge(dfr, gold_standard_df, on='sample', how='inner')

def calculate_agreement(merged_df):
    # Return the percentage of samples where gpt_generated_biome agrees with curated_biome
    total_samples = len(merged_df)
    agree_samples = len(merged_df[merged_df['gpt_generated_biome'] == merged_df['curated_biome']])
    return (agree_samples / total_samples) * 100

# Extract datetime for x-axis labeling
def extract_datetime_from_filename(file):
    match = re.search(r'(\d{8})_(\d{6})', file)
    if match:
        date, time = match.groups()
        return f"{date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:4]}:{time[4:]}"
    return "Unknown"

datetimes = [extract_datetime_from_filename(file) for file in all_files]

# Structure to store the agreement data
agreement_df = pd.DataFrame(columns=['datetime', 'curated_biome', 'agreement_percentage'])

# Process each file and calculate agreement percentages
for file in all_files:
    datetime_label = extract_datetime_from_filename(file)
    processed_df = load_and_process_file(file, gold_dict_df)
    print(len(processed_df))
    # Calculate agreement for each biome
    biome_agreement = processed_df.groupby('curated_biome').apply(lambda x: calculate_agreement(x)).reset_index()
    biome_agreement.columns = ['curated_biome', 'agreement_percentage']
    biome_agreement['datetime'] = datetime_label
    agreement_df = pd.concat([agreement_df, biome_agreement], ignore_index=True)


# Define custom colors
custom_palette = {
    'animal': 'pink',
    'water': 'blue',
    'plant': 'green',
    'soil': 'brown'
}

# Plotting
plt.figure(figsize=(15,10))

# Create a barplot with custom colors
sns.barplot(x='datetime', y='agreement_percentage', hue='curated_biome', data=agreement_df, ci=None, palette=custom_palette, edgecolor='black', alpha=0.7)

# Add horizontal dashed lines
for y in [20, 40, 60, 80, 100]:
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

# Adjusting aesthetics
plt.xticks(rotation=45, ha='right')
plt.xlabel('Datetime', fontsize=14)
plt.ylabel('Agreement Percentage', fontsize=14)
plt.title('Agreement of GPT predictions with Curated Biomes across Runs', fontsize=16)
plt.legend(title='Curated Biome', loc='upper right', fontsize=12)
plt.ylim(0, 110)  # to ensure the y-axis starts from 0 and gives some space above 100
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()

plt.show()


