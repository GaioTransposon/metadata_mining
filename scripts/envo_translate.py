#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:08:44 2023

@author: dgaio
"""

import pandas as pd


# Load the data from CSV
data = pd.read_csv('/Users/dgaio/Downloads/ENVO.tsv', sep='\t')

# Extract ENVO labels from the 'Term IRI' and 'Parent term IRI' columns
data['label'] = data['Term IRI'].str.split('/').str[-1]
data['Parent_label'] = data['Parent term IRI'].str.split('/').str[-1]

# Create a combined description column based on the presence of a definition
data['Joint_Info'] = data.apply(lambda row: f"{row['Term label']} (definition: {row['Definition']})" if pd.notna(row['Definition']) else row['Term label'], axis=1)

# Create new DataFrames for the child and parent labels, text-labels, and other columns
child_df = data[['label', 'Term label', 'Definition', 'Joint_Info']]
parent_df = data[['Parent_label', 'Parent term label']].rename(columns={'Parent_label': 'label', 'Parent term label': 'Term label'}).assign(Joint_Info=data['Parent term label'])

# Concatenate both DataFrames vertically
result = pd.concat([child_df, parent_df], axis=0, ignore_index=True)

# Drop duplicates based on 'label' and 'Term label' columns
result = result.drop_duplicates(subset=['label', 'Term label'])

# Keep only rows where the label starts with the specified patterns
patterns = ['ENVO_', 'NCBITaxon_', 'FOODON_', 'PO_', 'UBERON_']
mask = result['label'].str.startswith(tuple(patterns)).fillna(False)
result = result[mask]


label_info_dict = result.set_index('label')['Joint_Info'].to_dict()

print(label_info_dict)


# Save the final dataframe to a new CSV file
output_df.to_csv('/Users/dgaio/Downloads/ENVO_parsed.tsv', index=False)