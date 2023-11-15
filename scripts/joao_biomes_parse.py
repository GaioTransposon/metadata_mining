#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:59:58 2023

@author: dgaio
"""


import pandas as pd
import numpy as np 

# Path to your CSV file
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_keywords.csv'

# Read the CSV file
df_ori = pd.read_csv(file_path)


pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.width', 1000)        # Sets the maximum width of each row


# Select only columns 0, 1, and 8
df_ori = df_ori.iloc[:, [0, 1, 2, 4, 8]].copy()


# Extract the string after the dot in the first column and rename it to 'sample'
df_ori['0'] = df_ori.iloc[:, 0].apply(lambda x: x.split('.')[1] if '.' in x else x)

# Remove duplicate rows
df_ori = df_ori.drop_duplicates()

df = df_ori.copy()
print(df.tail())



# =============================================================================
# # Inspection time: have a look at rows with lots of biomes: 
# # Count the occurrences of "|" in each row of column 1
# pipe_count = df.iloc[:, 1].str.count('\|')
# 
# # Filter the DataFrame to show only rows with more than three "|"
# rows_with_more_than_three_pipes = df[pipe_count > 3]
# 
# print(rows_with_more_than_three_pipes)
# 
# =============================================================================

# =============================================================================
# # Inspection time: have a look at rows without biomes:
# # checking a few samples in the original df that did not have any string under biome. 
# # let's grab random n of these samples:
# nan_rows = df[df.iloc[:, 1].isna()]
# n = 20
# list_of_samples = nan_rows.sample(n=n).iloc[:, 0].tolist()
# 
# # look for those in the original df:
# filtered_df = df[df['0'].isin(list_of_samples)]
# 
# print(filtered_df)
# =============================================================================


def calculate_confidence(row):
    if not isinstance(row, str):
        return None, None, 'unknown'

    # Extract the left-most biome and sub-biome
    first_segment = row.split('|')[0]
    parts = first_segment.split(';')
    biome = parts[0]
    sub_biome = parts[1] if len(parts) > 1 else None

    # Count occurrences of each biome within the row
    row_biomes = [segment.split(';')[0] for segment in row.split('|')]
    biome_counts = {b: row_biomes.count(b) for b in set(row_biomes)}
    total_counts = sum(biome_counts.values())

    # Calculate the proportion for the left-most biome
    biome_proportion = biome_counts[biome] / total_counts if total_counts > 0 else 0

    # Determine confidence based on the proportion
    if biome_proportion >= 0.6:
        confidence = 'high'
    elif biome_proportion > 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'

    return biome, sub_biome, confidence

# Apply function
df['biome'], df['sub_biome'], df['confidence'] = zip(*df['1'].apply(calculate_confidence))

print(df)



# =============================================================================
# =============================================================================
# How we are obtaining the confidence score: 
# Example: aquatic;sea|soil;field|plant;rhizosphere|aquatic;marine|aquatic;sediment
# 
# Left-Most Biome: aquatic
# Sub-Biome: sea
# 
# Count Occurrences Within the Row:
# soil: 1 time
# plant: 1 time
# aquatic: 3 times (sea, marine, sediment)
# 
# Calculate Confidence:
# Total Biome Counts in the Row: 5 (soil 1 + plant 1 + aquatic 3)
# Proportion for the Left-Most Biome (aquatic): 3/5 = 0.6
# 
# Determine Confidence Based on Proportion:
# Confidence is 'high' if proportion ≥ 0.6
# Confidence is 'medium' if proportion > 0.4 and < 0.6
# Confidence is 'low' if proportion ≤ 0.4
# 
# In this example, the proportion for aquatic is 0.6, so the Confidence: High.
# =============================================================================
# =============================================================================

# Slice to inspect
sliced_df = df.iloc[1:1000, ]

# Count the number of rows for each confidence level
confidence_counts = df['confidence'].value_counts()
print('Counts per confidence interval:\n', confidence_counts)

df.rename(columns={'0': 'sample', '8': 'coordinates'}, inplace=True)

# Select and reorder the columns
df = df[['sample', 'biome', 'sub_biome', 'confidence', 'coordinates']]


# Create the biome_confidence column
df['biome_confidence'] = df.apply(lambda row: f"{row['biome']}_{row['confidence']}" 
                                  if pd.notna(row['biome']) else np.nan, axis=1)
print(df)


# Save to file
df.to_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/joao_biomes_parsed.csv', index=False)


























