#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:02:23 2024

@author: dgaio
"""





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_results.csv'
df = pd.read_csv(file_path)  # Adjust if your file format is different

# Retain and reorder specified columns, removing 'MWU Statistic'
columns_to_keep = [
    'Label', 
    'Agreement biome (exact match)', 
    'Agreement biome (lenient match)', 
    'Average Similarity', 
    'Median Similarity', 
    'Standard Deviation', 
    'MWU P-value'
]
df = df[columns_to_keep]

# Rename columns
column_rename_map = {
    'Agreement biome (exact match)': 'agreement_exact',
    'Agreement biome (lenient match)': 'agreement_lenient',
    'Average Similarity': 'avg_sim',
    'Median Similarity': 'med_sim',
    'Standard Deviation': 'SD',
    'MWU P-value': 'MWU_pval'
}
df.rename(columns=column_rename_map, inplace=True)

# Convert percentage columns to numeric and extract middle value from '95_perc'
df['agreement_exact'] = df['agreement_exact'].str.extract(r'(\d+\.\d+)%').astype(float)
df['agreement_lenient'] = df['agreement_lenient'].str.extract(r'(\d+\.\d+)%').astype(float)


#df = df[0:7] # for chunking y/n + chunk sizes
df = df[8:68]   # for creativity params

df.reset_index(drop=True, inplace=True)  # Reset the index to reflect the new order


# Record where NaN rows were before dropping them
nan_indices = df[df.isna().any(axis=1)].index.tolist()

# Drop rows with NaN values
df_cleaned = df.dropna()

# Adjust NaN indices to match positions in the cleaned DataFrame
adjusted_nan_indices = [i - sum(idx < i for idx in nan_indices) for i in nan_indices]

# Plotting each column with its respective color map
colors = ['Blues', 'Blues', 'Greens', 'Greens', 'Oranges', 'Greys']
fig_height = max(10, len(df_cleaned) * 0.3)  # Scale height by number of rows
# fig, axes = plt.subplots(nrows=1, ncols=len(df.columns)-1, figsize=(20, 5), sharey=True)
fig, axes = plt.subplots(nrows=1, ncols=len(df_cleaned.columns)-1, figsize=(20, fig_height), sharey=True)


custom_col_names = [
    'Agreement \n (exact match)',
    'Agreement \n (lenient match)',
    'Average \n similarity',
    'Median \n similarity',
    'Standard \n deviation',
    'MWU \n p-value'
]


# Plotting each column with its respective color map
for ax, col, cmap, display_name in zip(axes, df_cleaned.columns[1:], colors, custom_col_names):
    sns.heatmap(df_cleaned[[col]].astype(float), ax=ax, annot=True, fmt=".2f", cmap=cmap, cbar=False, annot_kws={"size": 6}) # values inside cells fontsize
    ax.set_title(display_name, fontdict={'fontsize': 10, 'fontweight': 'normal'})  # Change font size and weight of column headers    ax.tick_params(left=False)  # Remove ticks but keep labels

    ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
    ax.tick_params(axis='y', which='both', length=0)

    # Manually adjust the position of the subplots to bring them closer together
    pos = ax.get_position()  # Get the current position of the subplot
    new_pos = [pos.x0 - 0.01, pos.y0, pos.width * 1.1, pos.height]  # Adjust spacing
    ax.set_position(new_pos)  # Set the new position

# Set y-ticks and y-tick labels to match the data rows
axes[0].set_yticklabels(df_cleaned['Label'], rotation=0, fontsize=min(100 // len(df_cleaned), 10))


# Add thick horizontal lines where the NaN rows were, using the adjusted indices
for ax in axes:
    for adjusted_idx in adjusted_nan_indices:
        ax.axhline(y=adjusted_idx, color='black', linewidth=3)


# Remove space between columns and adjust layout
plt.subplots_adjust(left=0.9, wspace=0, hspace=0)  # Increase 'left' if labels are still cut off
plt.tight_layout()
plt.show()






from tabulate import tabulate

# Define headers
headers = df.columns

# Convert the DataFrame to a list of lists for tabulate
table = tabulate(df.values, headers, tablefmt="grid", missingval="")

# Print the formatted table
print(table)




