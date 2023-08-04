#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:52:19 2023

@author: dgaio
"""


import pandas as pd

# Assuming the data is in a CSV file named 'data.csv'
df = pd.read_csv('/Users/dgaio/Downloads/ENVO.tsv', sep='\t')

# Extract only the ENVO codes from the IRI columns
df['Term IRI'] = df['Term IRI'].apply(lambda x: x.split('/')[-1])
df['Parent term IRI'] = df['Parent term IRI'].apply(lambda x: x.split('/')[-1] if pd.notnull(x) else x)

# Separate the term and parent term into two separate dataframes, 
# and rename the columns to 'ENVO_Code' and 'Description'
df1 = df[['Term IRI', 'Term label']].rename(columns={'Term IRI': 'ENVO_code', 'Term label': 'Description'})
df2 = df[['Parent term IRI', 'Parent term label']].rename(columns={'Parent term IRI': 'ENVO_code', 'Parent term label': 'Description'})

# Concatenate the dataframes vertically
output_df = pd.concat([df1, df2], ignore_index=True)

# Drop the rows where any column has NaN
output_df = output_df.dropna()

# Remove duplicate rows
output_df = output_df.drop_duplicates()


# Get all unique ENVO_Code
unique_codes = output_df['ENVO_code'].unique()

# Extract prefix of each unique code
prefixes = set(code.split('_')[0] for code in unique_codes)

print(prefixes)



# Save the final dataframe to a new CSV file
output_df.to_csv('/Users/dgaio/Downloads/ENVO_parsed.tsv', index=False)

