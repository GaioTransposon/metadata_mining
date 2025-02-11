#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:48:46 2025

@author: danielagaio
"""


import pandas as pd

# Load the Excel file
df = pd.read_excel('/Users/danielagaio/github/metadata_mining/middle_dir/GH_collect_output_here.xlsx')

# Format the 'location' column by enclosing in quotes
df['location'] = '"' + df['location'].astype(str) + '"'

# Format the 'keywords (comma-separated)' column by enclosing in curly braces
df['keywords (comma-separated)'] = '{' + df['keywords (comma-separated)'].str.replace(',', ', ') + '}'

# Select and rename columns
df = df[['sample ID', 'biome', 'location', 'keywords (comma-separated)', 'description']]
df.columns = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4']

# Save to txt file
df.to_csv('/Users/danielagaio/github/metadata_mining/middle_dir/GH_collect_output_here.txt', index=False, header=True, sep=',', quoting=2)
