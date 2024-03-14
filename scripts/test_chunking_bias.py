#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:53:10 2024

@author: dgaio
"""


import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt



def parse_metadata(file_path):
    batch_number = 0
    data = []
    position_within_batch = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if '-----' in line and position_within_batch > 0:
                batch_number += 1
                position_within_batch = 0
            elif 'sample_ID=' in line and '>' in line:
                sample_id = line.split('>')[1].strip()
                position_within_batch += 1
                data.append((sample_id, f'batch {batch_number + 1}', position_within_batch))

    return pd.DataFrame(data, columns=['Sample ID', 'Batch Number', 'Position Within Batch'])

def merge_with_biome_info(df_filtered, file_path_biome_info):
    df_biome_info = pd.read_csv(file_path_biome_info)
    df_merged = pd.merge(df_filtered, df_biome_info[['col_0', 'col_1']], left_on='Sample ID', right_on='col_0', how='left')
    df_merged.rename(columns={'col_1': 'Biome'}, inplace=True)
    df_merged.drop(columns=['col_0'], inplace=True)
    return df_merged

def process_files(metadata_file_path, biome_info_file_path):
    df_metadata = parse_metadata(metadata_file_path)

    batch_counts = df_metadata['Batch Number'].value_counts()
    large_batches = batch_counts[batch_counts >= 5].index.tolist()
    df_filtered = df_metadata[df_metadata['Batch Number'].isin(large_batches)]

    df_final = merge_with_biome_info(df_filtered, biome_info_file_path)
    return df_final

def categorize_position(row):
    total = row['Batch Size']
    position = row['Position Within Batch']
    beginning_size = total // 3 if total % 3 == 0 else total // 3 + 1
    end_start = total - beginning_size + 1

    if position <= beginning_size:
        return 'beginning'
    elif position >= end_start:
        return 'end'
    else:
        return 'middle'


def sample_equal_number(group, seed):
    return group.sample(n=min_group_size, random_state=seed)




    
# List of metadata and GPT output file pairs
file_pairs = [
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131529.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1536.txt"),
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131617.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1624.txt"),
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131733.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1740.txt")
]


# Load the gold dictionary
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")




with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict = gold_dict[0]
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['gold_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)

# Define seeds for different runs
seeds = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
final_results_df = pd.DataFrame()  # Initialize the final DataFrame to store results from all runs

for seed in seeds:
    dfs = []
    for metadata_file, biome_info_file in file_pairs:
        dfs.append(process_files(metadata_file, biome_info_file))

    final_df = pd.concat(dfs, ignore_index=True)
    final_df_with_gold = pd.merge(final_df, gold_dict_df[['sample', 'gold_biome']], left_on='Sample ID', right_on='sample', how='left')
    final_df_with_gold.drop(columns='sample', inplace=True)

# =============================================================================
#     ###
#     # filtering out "other"
#     final_df_with_gold = pd.merge(final_df, gold_dict_df[['sample', 'gold_biome']], left_on='Sample ID', right_on='sample', how='left')
#     final_df_with_gold.drop(columns='sample', inplace=True)
#     
#     # Filter out samples with the biome "other"
#     final_df_with_gold = final_df_with_gold[final_df_with_gold['Biome'] == 'other']
#     
#     # Add correctness column
#     final_df_with_gold['Correct Guess'] = final_df_with_gold['Biome'] == final_df_with_gold['gold_biome']
#     ###
# =============================================================================


    final_df_with_gold['Correct Guess'] = final_df_with_gold['Biome'] == final_df_with_gold['gold_biome']

    batch_sizes = final_df_with_gold['Batch Number'].value_counts().rename('Batch Size')
    final_df_with_gold = final_df_with_gold.merge(batch_sizes, left_on='Batch Number', right_index=True)

    final_df_with_gold['Position Category'] = final_df_with_gold.apply(categorize_position, axis=1)

    min_group_size = final_df_with_gold.groupby('Position Category').size().min()

    sampled_df = final_df_with_gold.groupby('Position Category').apply(lambda x: sample_equal_number(x, seed)).reset_index(drop=True)

    final_results_df = pd.concat([final_results_df, sampled_df], ignore_index=True)

print(final_results_df)



# how many beg, mid, end, do I have? 
category_counts = final_results_df['Position Category'].value_counts()

# Print the counts
print(category_counts)



# plot

# Prepare data: count occurrences of correct/incorrect within each position category
grouped_data = final_results_df.groupby(['Position Category', 'Correct Guess']).size().unstack(fill_value=0)

# Plot
fig, ax = plt.subplots()

# Define the bar width
bar_width = 0.35

# Positions of the bars on the x-axis
r1 = range(len(grouped_data))

# Plotting the bars
correct_bars = ax.bar(r1, grouped_data[True], color='green', edgecolor='black', width=bar_width, label='Correct')
incorrect_bars = ax.bar(r1, grouped_data[False], bottom=grouped_data[True], color='red', edgecolor='black', width=bar_width, label='Incorrect')

# Add text for labels, title, and axes ticks
ax.set_xlabel('position of sample within batch')
ax.set_ylabel('# samples')
ax.set_title('correctedness of prediction by GPT - *all biomes*') # ("other" only)
ax.set_xticks(r1)
ax.set_xticklabels(['beginning', 'middle', 'end'])

# Adding the legend and showing the plot
ax.legend()
plt.show()















