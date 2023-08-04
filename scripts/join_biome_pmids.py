#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:19:22 2023

@author: dgaio
"""

# run as: 
# python ~/github/metadata_mining/scripts/join_biome_pmids.py --df1 "cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv" --df2 "cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid.csv" --output_file "cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid_biome.csv" --output_figure "cloudstor/Gaio/MicrobeAtlasProject/plot_sample.info_pmid_biome.pdf" 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import argparse  
import os 

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process and visualize data.')
parser.add_argument('--df1', type=str, required=True, help='Give path to otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv')
parser.add_argument('--df2', type=str, required=True, help='Give path to sample.info_pmid.csv')
parser.add_argument('--output_file', type=str, required=True, help='Give path to and name of output file')
parser.add_argument('--output_figure', type=str, default=None, help='Optional output figure path. If provided, the plot will be saved as a PDF at this location.')
args = parser.parse_args()



home = os.path.expanduser('~')

df1_path = os.path.join(home, args.df1)
df2_path = os.path.join(home, args.df2)
output_path = os.path.join(home, args.output_file)

df1 = pd.read_csv(df1_path, sep='\t')
df2 = pd.read_csv(df2_path)


# Extract the sample names from the first dataframe
df1['sample'] = df1['SampleID'].str.split('.').str[1]

# Set the number of lines to handle in df1 and df2
n_lines_df1 = len(df1)  # Change this to the desired number of lines
n_lines_df2 = len(df2)  # Change this to the desired number of lines

# Slice the dataframes to the specified number of lines
df1_sliced = df1[:n_lines_df1]
df2_sliced = df2[:n_lines_df2]


# Merge the dataframes based on the common sample names
merged_df = pd.merge(df1, df2, left_on='sample', right_on='sample')

# Drop useless columns 
merged_df.drop('SampleID', axis=1, inplace=True)
merged_df.drop('pmid', axis=1, inplace=True)


# save file: 
merged_df.to_csv(output_path, index=False)

###########################################################################
###########################################################################
###########################################################################


# Visualize: 

df_nan = merged_df['pmid_digits'].isna().groupby(merged_df['EnvClean_merged']).sum().astype(int).reset_index(name='NaN')
df_not_nan = merged_df['pmid_digits'].notna().groupby(merged_df['EnvClean_merged']).sum().astype(int).reset_index(name='Not NaN')


df = pd.merge(df_nan, df_not_nan, on='EnvClean_merged')

labels = df['EnvClean_merged'].values
nan_values = df['NaN'].values
not_nan_values = df['Not NaN'].values

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, nan_values, width, label='NaN')
rects2 = ax.bar(x + width/2, not_nan_values, width, label='Not NaN')

ax.set_xlabel('Biome')
ax.set_title('Number of NaN and non-NaN values by Biome')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

if args.output_figure:
    figure_path = os.path.join(home, args.output_figure)
    plt.savefig(figure_path, format='pdf')

plt.show()




