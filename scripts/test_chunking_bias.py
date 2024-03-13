#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:53:10 2024

@author: dgaio
"""


import pandas as pd

def process_files(metadata_file_path, biome_info_file_path):
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

    # Parse the metadata file
    df_metadata = parse_metadata(metadata_file_path)

    # Filter for batches with 5 or more samples
    batch_counts = df_metadata['Batch Number'].value_counts()
    large_batches = batch_counts[batch_counts >= 5].index.tolist()
    df_filtered = df_metadata[df_metadata['Batch Number'].isin(large_batches)]

    # Merge with biome information and return the result
    df_final = merge_with_biome_info(df_filtered, biome_info_file_path)
    return df_final

# List of metadata and GPT output file pairs
file_pairs = [
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131529.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1536.txt"),
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131617.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1624.txt")#,
    # Add more file pairs as needed
]

# Process each pair and concatenate the results
dfs = []
for metadata_file, biome_info_file in file_pairs:
    dfs.append(process_files(metadata_file, biome_info_file))

# Concatenate all resulting DataFrames
final_df = pd.concat(dfs, ignore_index=True)

# Display the final, comprehensive DataFrame
print(final_df)







