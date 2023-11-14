#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:59:58 2023

@author: dgaio
"""


import pandas as pd

# Path to your CSV file
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_keywords.csv'

# Read the CSV file
df = pd.read_csv(file_path)


pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.width', 1000)        # Sets the maximum width of each row


# Display the first few rows of the DataFrame
print(df.tail())


# Create 'sample' column
df['sample'] = df.iloc[:, 0].apply(lambda x: x.split('.')[1] if '.' in x else x)

# Split the '1' column on "|"
split_df = df.iloc[:, 1].str.split('|', expand=True)

# Determine the maximum number of splits
max_splits = split_df.shape[1]

# Generate dynamic column names
column_names = [f'Joao_biome_{i+1}' for i in range(max_splits)]

# Replace the columns in split_df with new names
split_df.columns = column_names


# Print the head of the DataFrame to verify
print(split_df.head())



















# Further split the new columns
df[['Joao_biome_1', 'Joao_subbiome_1']] = df['Joao_biome_1'].str.split(';', expand=True)
df[['Joao_biome_2', 'Joao_subbiome_2']] = df['Joao_biome_2'].str.split(';', expand=True)

# Fill NA in Joao_biome_2 and its sub-columns
df['Joao_biome_2'].fillna('NA', inplace=True)
df['Joao_subbiome_2'].fillna('NA', inplace=True)

# Rename column 8 to "coordinates"
df.rename(columns={8: 'coordinates'}, inplace=True)

# Print the head of the DataFrame
print(df.head())
