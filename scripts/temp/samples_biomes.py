#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:08:15 2023

@author: dgaio
"""


# run as: 
# python ~/github/metadata_mining/scripts/samples_biomes.py --jankos_file "cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv" --output_file "cloudstor/Gaio/MicrobeAtlasProject/samples_biomes.csv" --output_figure "cloudstor/Gaio/MicrobeAtlasProject/plot_samples_biomes.pdf" 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import argparse  
import os 

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process and visualize data.')
parser.add_argument('--jankos_file', type=str, required=True, help='Give path to otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv')
parser.add_argument('--output_file', type=str, required=True, help='Give path to and name of output file')
parser.add_argument('--output_figure', type=str, default=None, help='Optional output figure path. If provided, the plot will be saved as a PDF at this location.')
args = parser.parse_args()



home = os.path.expanduser('~')

jankos_file_path = os.path.join(home, args.jankos_file)
output_path = os.path.join(home, args.output_file)

jankos_df = pd.read_csv(jankos_file_path, sep='\t')


# Extract the sample names from the first dataframe
jankos_df['sample'] = jankos_df['SampleID'].str.split('.').str[1]

# Drop useless columns 
jankos_df.drop('SampleID', axis=1, inplace=True)

# Rename first column
jankos_df = jankos_df.rename(columns={"EnvClean_merged": "biome"})

# save file: 
jankos_df.to_csv(output_path, index=False)

###########################################################################
###########################################################################
###########################################################################


# Function to extract non-digit characters
def get_alpha(s):
    return ''.join(c for c in s if not c.isdigit())

# Apply function to 'sample' column and get unique values
unique_alpha = jankos_df['sample'].apply(get_alpha).unique()
print('These sample names are preceded by the following unique characters: ', unique_alpha)








# Prepare color mapping
colors = {
    'animal': 'red',
    'soil': 'brown',
    'aquatic': 'blue',
    'plant': 'green',
    'unknown': 'gray'
}

# Count number of samples per biome
biome_counts = jankos_df['biome'].value_counts()

# Apply color mapping to biome types
bar_colors = [colors.get(biome, 'gray') for biome in biome_counts.index]

# Create bar plot
plt.figure(figsize=(10, 5))
bars = plt.bar(biome_counts.index, biome_counts.values, color=bar_colors)
plt.xlabel('Biome')
plt.ylabel('Sample Count')
plt.title('Number of samples per biome')

# Add counts on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment

plt.show()




# =============================================================================
# # Visualize: 
# 
# df_nan = merged_df['pmid_digits'].isna().groupby(merged_df['EnvClean_merged']).sum().astype(int).reset_index(name='NaN')
# df_not_nan = merged_df['pmid_digits'].notna().groupby(merged_df['EnvClean_merged']).sum().astype(int).reset_index(name='Not NaN')
# 
# 
# df = pd.merge(df_nan, df_not_nan, on='EnvClean_merged')
# 
# labels = df['EnvClean_merged'].values
# nan_values = df['NaN'].values
# not_nan_values = df['Not NaN'].values
# 
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
# 
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, nan_values, width, label='NaN')
# rects2 = ax.bar(x + width/2, not_nan_values, width, label='Not NaN')
# 
# ax.set_xlabel('Biome')
# ax.set_title('Number of NaN and non-NaN values by Biome')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# 
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# 
# fig.tight_layout()
# 
# if args.output_figure:
#     figure_path = os.path.join(home, args.output_figure)
#     plt.savefig(figure_path, format='pdf')
# 
# plt.show()
# 
# =============================================================================
