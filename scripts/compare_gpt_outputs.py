#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:05:41 2023

@author: dgaio
"""

# =============================================================================
# import glob
# import pandas as pd
# import os 
# import openai
# import pandas as pd
# import numpy as np  
# from collections import Counter
# import argparse  
# import pickle
# import re
# from datetime import datetime
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# 
# 
# 
# 
# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
# input_df = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")
# 
# 
# # 1. Load all files
# file_pattern = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean*"
# all_files = glob.glob(file_pattern)
# 
# 
# # Your gold standard df, for example purposes
# input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
# 
# ########### # 2. open gold_dict and transform to a df
# with open(input_gold_dict, 'rb') as file:
#     input_gold_dict = pickle.load(file)
# 
# input_gold_dict = input_gold_dict[0] # this is because the second item is the list of pmids I processed - it was necessary to run confirm_biome_game.py, now not necessary anymore 
# gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
# gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
# gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
# gold_dict_df.drop(columns='tuple_data', inplace=True)
# 
# 
# 
# 
# # Assuming gold_standard_df exists with columns 'sample' and 'curated_biome'
# # Assuming all_files list has been populated
# 
# def load_and_process_file(file_name, gold_standard_df):
#     dfr = pd.read_csv(file_name)
#     return pd.merge(dfr, gold_standard_df, on='sample', how='inner')
# 
# def calculate_agreement(merged_df):
#     # Return the percentage of samples where gpt_generated_output_clean agrees with curated_biome
#     total_samples = len(merged_df)
#     agree_samples = len(merged_df[merged_df['gpt_generated_output_clean'] == merged_df['curated_biome']])
#     return (agree_samples / total_samples) * 100
# 
# 
# def extract_datetime_and_temp_from_filename(file):
#     date_match = re.search(r'(\d{8})_(\d{6})', file)
#     temp_match = re.search(r'temp([\d\.]+)_', file)
#     date_str, time_str, temp_str = "Unknown", "Unknown", "Unknown"
#     if date_match:
#         date, time = date_match.groups()
#         date_str = f"{date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:4]}:{time[4:]}"
#     if temp_match:
#         temp_str = temp_match.group(1)
#     return date_str, temp_str
# 
# datetimes_and_temps = [extract_datetime_and_temp_from_filename(file) for file in all_files]
# 
# 
# 
# 
# 
# 
# datetimes = [extract_datetime_from_filename(file) for file in all_files]
# 
# # Structure to store the agreement data
# agreement_df = pd.DataFrame(columns=['datetime', 'curated_biome', 'agreement_percentage'])
# 
# # Process each file and calculate agreement percentages
# for file, (datetime_label, temp_label) in zip(all_files, datetimes_and_temps):
#     biome_agreement['temp'] = temp_label
#     
#     datetime_label = extract_datetime_from_filename(file)
#     processed_df = load_and_process_file(file, gold_dict_df)
#     print(len(processed_df))
#     # Calculate agreement for each biome
#     biome_agreement = processed_df.groupby('curated_biome').apply(lambda x: calculate_agreement(x)).reset_index()
#     biome_agreement.columns = ['curated_biome', 'agreement_percentage']
#     biome_agreement['datetime'] = datetime_label
#     agreement_df = pd.concat([agreement_df, biome_agreement], ignore_index=True)
# 
# 
# # Define custom colors
# custom_palette = {
#     'animal': 'pink',
#     'water': 'blue',
#     'plant': 'green',
#     'soil': 'brown'
# }
# 
# # Plotting
# plt.figure(figsize=(15,10))
# 
# # Create a barplot with custom colors
# sns.barplot(x='datetime', y='agreement_percentage', hue='curated_biome', data=agreement_df, ci=None, palette=custom_palette, edgecolor='black', alpha=0.7)
# # Create a barplot with custom colors
# sns.barplot(x='temp', y='agreement_percentage', hue='curated_biome', data=agreement_df, ci=None, palette=custom_palette, edgecolor='black', alpha=0.7)
# 
# # Add horizontal dashed lines
# for y in [20, 40, 60, 80, 100]:
#     plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
# 
# # Adjusting aesthetics
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Datetime', fontsize=14)
# plt.ylabel('Agreement Percentage', fontsize=14)
# plt.title('Agreement of GPT predictions with Curated Biomes across Runs', fontsize=16)
# plt.legend(title='Curated Biome', loc='upper right', fontsize=12)
# plt.ylim(0, 110)  # to ensure the y-axis starts from 0 and gives some space above 100
# plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
# plt.tight_layout()
# 
# plt.show()
# =============================================================================


import glob
import pandas as pd
import os 
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns

# For testing purposes
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"

# 1. Load all files
file_pattern = os.path.join(work_dir, "gpt_clean*")
all_files = glob.glob(file_pattern)
print(all_files)

# 2. Open gold_dict and transform to a df
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict = gold_dict[0]
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)

def load_and_process_file(file_name, gold_standard_df):
    dfr = pd.read_csv(file_name)
    return pd.merge(dfr, gold_standard_df, on='sample', how='inner')

def calculate_agreement(merged_df):
    total_samples = len(merged_df)
    agree_samples = len(merged_df[merged_df['gpt_generated_output_clean'] == merged_df['curated_biome']])
    return (agree_samples / total_samples) * 100

def extract_datetime_and_temp_from_filename(file):
    # Adjusted the date_match regex to account for '_dt' between date and time and for the HHMM format
    match = re.search(r'temp([\d\.]+)_dt(\d{8})_(\d{4})', file)
    date_str = "Unknown"
    temp_str = "Unknown"
    if match:
        temp_str, date, time = match.groups()
        date_str = f"{date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:]}"  # Only hours and minutes
    return date_str, temp_str



datetimes_and_temps = [extract_datetime_and_temp_from_filename(file) for file in all_files]

agreement_df = pd.DataFrame(columns=['datetime', 'temp', 'curated_biome', 'agreement_percentage'])

# Process each file and calculate agreement percentages
for file, (datetime_label, temp_label) in zip(all_files, datetimes_and_temps):
    processed_df = load_and_process_file(file, gold_dict_df)
    
    # Calculate agreement for each biome
    biome_agreement = processed_df.groupby('curated_biome').apply(lambda x: calculate_agreement(x)).reset_index()
    biome_agreement.columns = ['curated_biome', 'agreement_percentage']
    biome_agreement['datetime'] = datetime_label
    biome_agreement['temp'] = temp_label
    
    agreement_df = pd.concat([agreement_df, biome_agreement], ignore_index=True)


# Sort the dataframe by 'temp' values
agreement_df['temp'] = agreement_df['temp'].astype(float)  # Convert to float to ensure proper sorting
agreement_df = agreement_df.sort_values(by='temp')

print(agreement_df)

# Improved custom colors
custom_palette = {
    'animal': '#E28989',  # light_coral
    'water': '#8ADAE1',   # tiffany_blue
    'plant': '#BCD279',   # pistachio
    'soil': '#DECE8A'     # flax
}

# Improved plotting
plt.figure(figsize=(18, 12))
sns.set_style("whitegrid")  # Setting a white grid background
sns.barplot(x='temp', y='agreement_percentage', hue='curated_biome', data=agreement_df, 
            ci=None, palette=custom_palette, edgecolor='black', alpha=0.85)

# Add horizontal dashed lines with improved style
for y in [20, 40, 60, 80, 100]:
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

# Adjusting aesthetics with improved font styles
plt.xticks(rotation=45, ha='right', fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")
plt.xlabel('Temperature', fontsize=16, fontweight='bold', fontname="Arial")
plt.ylabel('Agreement Percentage', fontsize=16, fontweight='bold', fontname="Arial")
plt.title('Agreement of GPT Predictions with Curated Biomes across Runs', fontsize=20, fontweight='bold', fontname="Arial")
plt.legend(title='Curated Biome', loc='upper right', fontsize=12, title_fontsize=14)
plt.ylim(0, 110)  # to ensure the y-axis starts from 0 and gives some space above 100
plt.tight_layout()

plt.show()
















