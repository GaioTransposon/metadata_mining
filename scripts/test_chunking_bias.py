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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def sample_equal_number(group, seed, min_group_size):
    return group.sample(n=min_group_size, random_state=seed)


def sample_by_position_category(df, seed):
    min_group_size = df.groupby('Position Category').size().min()
    return df.groupby('Position Category').apply(lambda x: sample_equal_number(x, seed, min_group_size)).reset_index(drop=True)



def load_gold_dict(input_path):
    with open(input_path, 'rb') as file:
        gold_dict = pickle.load(file)[0]
    gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
    gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
    gold_dict_df['gold_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
    gold_dict_df.drop(columns='tuple_data', inplace=True)
    return gold_dict_df

def merge_with_gold_df(df, gold_df):
    merged_df = pd.merge(df, gold_df[['sample', 'gold_biome']], left_on='Sample ID', right_on='sample', how='left')
    merged_df.drop(columns='sample', inplace=True)
    merged_df['Correct Guess'] = merged_df['Biome'] == merged_df['gold_biome']
    return merged_df

def filter_by_biome(df, biome):
    if biome:
        return df[df['Biome'] == biome]
    return df

def add_batch_info(df):
    batch_sizes = df['Batch Number'].value_counts().rename('Batch Size')
    df = df.merge(batch_sizes, left_on='Batch Number', right_index=True)
    df['Position Category'] = df.apply(categorize_position, axis=1)
    return df


def plot_results(df, biome_to_select, save_plot_path):
    grouped_data = df.groupby(['Position Category', 'Correct Guess']).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    bar_width = 0.35
    r1 = range(len(grouped_data))

    ax.bar(r1, grouped_data[True], color='green', edgecolor='black', width=bar_width, label='Correct')
    ax.bar(r1, grouped_data[False], bottom=grouped_data[True], color='red', edgecolor='black', width=bar_width, label='Incorrect')

    ax.set_xlabel('position of sample within batch')
    ax.set_ylabel('# samples')
    title_biome = "all biomes" if not biome_to_select else biome_to_select
    ax.set_title(f'correctness of prediction by position - {title_biome}')
    ax.set_xticks(r1)
    ax.set_xticklabels(['beg', 'mid', 'end'])

    ax.legend()

    # Save the plot using the title as the file name
    file_name = f"{title_biome}.png"
    full_path = os.path.join(save_plot_path, file_name)
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

    plt.show()  
    
    
def plot_histogram(df, biome_to_select, save_plot_path):
    # Group by 'Position Within Batch' and count occurrences
    grouped_data = df['Position Within Batch'].value_counts().sort_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Creating the histogram
    plt.bar(grouped_data.index, grouped_data.values, color='blue', width=0.4)

    # Enhancing the plot
    plt.xlabel('position within batch')
    plt.ylabel('# samples')
    title_biome = "all biomes" if not biome_to_select else biome_to_select
    
    # Save the plot using the title as the file name
    file_name = f"occurrences_by_position_{title_biome}.png"
    full_path = os.path.join(save_plot_path, file_name)
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

    plt.tight_layout()
    plt.show()



    



file_pairs = [
    
    # 1500
    #("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403191312.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API290_dt20240319_1319.txt")
    
    # 2000
    #("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403191349.txt", "gpt_clean_output_nspb200_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API221_dt20240319_1358.txt") 
    
    # 3000
    #("metadata_chunks_202403191401.txt", "gpt_clean_output_nspb200_chunksize3000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API141_dt20240319_1408.txt")
    
    # 4000
    #("metadata_chunks_202403191411.txt", "gpt_clean_output_nspb200_chunksize4000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API104_dt20240319_1418.txt")
    
    # 5000
    #("metadata_chunks_202403191424.txt", "gpt_clean_output_nspb200_chunksize5000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API82_dt20240319_1432.txt")
    
    # 6000
    #("metadata_chunks_202403191459.txt", "gpt_clean_output_nspb200_chunksize6000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API68_dt20240319_1506.txt")
    
    # 8000
    ("metadata_chunks_202403191641.txt", "gpt_clean_output_nspb200_chunksize8000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs42_API51_dt20240319_1647.txt")
    
    #("metadata_chunks_202403181920.txt", "gpt_clean_output_nspb100_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs88_API109_dt20240318_1920.txt") 
    #("metadata_chunks_202403181902.txt", "gpt_clean_output_nspb100_chunksize4000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs88_API52_dt20240318_1902.txt")
    #("metadata_chunks_202403181848.txt", "gpt_clean_output_nspb100_chunksize6000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_rs88_API34_dt20240318_1848.txt")
    
    #("metadata_chunks_202403131529.txt", "gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1536.txt"),
    #("metadata_chunks_202403131617.txt", "gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1624.txt"),
    #("metadata_chunks_202403131733.txt", "gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1740.txt")
]


# List of metadata and GPT output file pairs
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
gold_dict_df = load_gold_dict(input_gold_dict)

# Specify the directory where you want to save the plot
save_plot_path = "/Users/dgaio/Desktop"

seed = 42
biome_to_select = ''  # Set to None or an empty string to make it optional

final_results_df = pd.DataFrame()

for metadata_file, biome_info_file in file_pairs:
    metadata_file_path = os.path.join(work_dir, metadata_file)
    biome_info_file_path = os.path.join(work_dir, biome_info_file)
    
    
    process_df = process_files(metadata_file_path, biome_info_file_path)

    merged_df = merge_with_gold_df(process_df, gold_dict_df)
    filtered_df = filter_by_biome(merged_df, biome_to_select)
    
    batch_info_df = add_batch_info(filtered_df)
    sampled_df = sample_by_position_category(batch_info_df, seed)
    final_results_df = pd.concat([final_results_df, sampled_df], ignore_index=True)

print(final_results_df['Position Category'].value_counts())

# Call plot_results with the additional save_path argument
plot_results(final_results_df, biome_to_select, save_plot_path)

plot_histogram(final_results_df, biome_to_select, save_plot_path)





























