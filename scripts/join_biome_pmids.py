#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:19:22 2023

@author: dgaio
"""

import pandas as pd
import time

df1 = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv', sep='\t')


df2 = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid.csv')

# Extract the sample names from the first dataframe
df1['sample'] = df1['SampleID'].str.split('.').str[1]

# Set the number of lines to handle in df1 and df2
n_lines_df1 = 40000  # Change this to the desired number of lines
n_lines_df2 = 40000  # Change this to the desired number of lines

# Slice the dataframes to the specified number of lines
df1_sliced = df1[:n_lines_df1]
df2_sliced = df2[:n_lines_df2]

# Start the timer for df1
start_time_df1 = time.time()

# Your code to handle df1_sliced goes here...

# Calculate the elapsed time for df1
elapsed_time_df1 = time.time() - start_time_df1

# Start the timer for df2
start_time_df2 = time.time()

# Merge the dataframes based on the common sample names
merged_df = pd.merge(df1, df2, left_on='sample', right_on='sample')

# Drop cols
merged_df.drop('SampleID', axis=1, inplace=True)
merged_df.drop('pmid', axis=1, inplace=True)

# Calculate the elapsed time for df2
elapsed_time_df2 = time.time() - start_time_df2

# Print the elapsed time for each dataframe
print(f"Elapsed Time for df1 ({n_lines_df1} lines): {elapsed_time_df1} seconds")
print(f"Elapsed Time for df2 ({n_lines_df2} lines): {elapsed_time_df2} seconds")



# visualize: 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

data = merged_df


df_nan = data['pmid_digits'].isna().groupby(data['EnvClean_merged']).sum().astype(int).reset_index(name='NaN')
df_not_nan = data['pmid_digits'].notna().groupby(data['EnvClean_merged']).sum().astype(int).reset_index(name='Not NaN')

df = pd.merge(df_nan, df_not_nan, on='EnvClean_merged')

labels = df['EnvClean_merged'].values
nan_values = df['NaN'].values
not_nan_values = df['Not NaN'].values

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, nan_values, width, label='NaN')
rects2 = ax.bar(x + width/2, not_nan_values, width, label='Not NaN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Biome')
ax.set_title('Number of NaN and non-NaN values by Biome')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()



