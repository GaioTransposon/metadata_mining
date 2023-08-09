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
import json




# for testing purposes 
large_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info'

biomes_df = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv' 
pmids_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid' 
dois_pmids_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_doi' 
bioprojects_pmcid_dict_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_bioproject' 

output_file = "sample.info_pmid_doi_biome.csv"



# open and transform also to dictionary: 
biomes_df = pd.read_csv(biomes_df, sep='\t')

# some df parse and col rename
biomes_df['sample'] = biomes_df['SampleID'].str.split('.').str[1]
biomes_df.drop('SampleID', axis=1, inplace=True)
biomes_df = biomes_df.rename(columns={'EnvClean_merged': 'biome'})
                        
    



# Open JSON files and load them as dictionaries
with open(pmids_dict_path, 'r') as pmids_file:
    pmids_dict = json.load(pmids_file)

with open(dois_pmids_dict_path, 'r') as dois_pmids_file:
    dois_pmids_dict = json.load(dois_pmids_file)
    
with open(bioprojects_pmcid_dict_path, 'r') as bioprojects_pmcid_dict_file:
    bioprojects_pmcid_dict = json.load(bioprojects_pmcid_dict_file)

# Now you can use pmids_dict and dois_pmids_dict as dictionaries
print(pmids_dict)
print(dois_pmids_dict)
print(bioprojects_pmcid_dict)










import pandas as pd

# =============================================================================
# # Given DataFrames and dictionaries
# biomes_df = pd.DataFrame({
#     'biome': ['animal', 'animal', 'soil'],
#     'sample': ['SRS7869555', 'SRS295012', 'SRS957269']
# })
# 
# pmids_dict = {
#     'SRS7869555': ['29554137'],
#     'SRS7869556': ['32042167'],
#     'SRS7869558': ['32042167']
# }
# 
# dois_pmids_dict = {
#     'SRS7869555': ['doi:10.1038/s41598-020-64819-2', '29554137'],
#     'SRS7698461': ['doi:10.1038/s41598-020-64819-2']
# }
# =============================================================================

# Convert pmids_dict to DataFrame
pmids_df = pd.DataFrame(list(pmids_dict.items()), columns=['sample', 'pmids'])
pmids_df['pmids'] = pmids_df['pmids'].apply(lambda x: ';'.join(x))

# Convert dois_pmids_dict to DataFrame, splitting doi and pmids
dois_pmids_df = pd.DataFrame(list(dois_pmids_dict.items()), columns=['sample', 'dois_pmids'])
dois_pmids_df['dois'] = dois_pmids_df['dois_pmids'].apply(lambda x: ';'.join([i for i in x if i.startswith('doi:')]))
dois_pmids_df['pmids_from_dois'] = dois_pmids_df['dois_pmids'].apply(lambda x: ';'.join([i for i in x if not i.startswith('doi:')]))
dois_pmids_df.drop(columns=['dois_pmids'], inplace=True)

# Merge the dataframes
final_df = biomes_df.merge(pmids_df, on='sample', how='left')\
                    .merge(dois_pmids_df, on='sample', how='left')

# Combine pmids and pmids_from_dois into a single column
final_df['all_pmids'] = final_df.apply(lambda row: ';'.join(filter(bool, [str(row['pmids']) if pd.notna(row['pmids']) else '', str(row['pmids_from_dois']) if pd.notna(row['pmids_from_dois']) else ''])), axis=1)

# Drop the separate pmids columns
final_df.drop(columns=['pmids', 'pmids_from_dois'], inplace=True)

# Handle NaN values if any
final_df.fillna('', inplace=True)

print(final_df)





# Count non-empty dois per biome
doi_counts = final_df[final_df['dois'] != ''].groupby('biome').size()

# Count non-empty all_pmids per biome
pmid_counts = final_df[final_df['all_pmids'] != ''].groupby('biome').size()

# Combine both counts into a DataFrame
counts_df = pd.DataFrame({'DOI Counts': doi_counts, 'PMID Counts': pmid_counts}).fillna(0)



import matplotlib.pyplot as plt

counts_df.plot(kind='bar', figsize=(10, 6))

ax = counts_df.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Counts')
plt.title('Non-Empty DOIs and PMIDs Counts per Biome')
plt.xticks(rotation=45)

# Loop through the bars and add the counts as text on top
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()






# Count the unique PMIDs per biome
unique_pmids_count = final_df[final_df['all_pmids'] != ''].groupby('biome')['all_pmids'].apply(lambda x: len(set(';'.join(x).split(';')))).reset_index(name='unique_pmids')

# Merge with the existing counts
counts_df = counts_df.merge(unique_pmids_count, on='biome', how='left').fillna(0)

# Plot
ax = counts_df.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Counts')
plt.title('Non-Empty DOIs, PMIDs, and Unique PMIDs Counts per Biome')
plt.xticks(rotation=45)

# Loop through the bars and add the counts as text on top
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()




















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









