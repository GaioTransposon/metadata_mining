#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:00:28 2024

@author: dgaio
"""

import pandas as pd
import numpy as np
from io import StringIO
import csv

# Define the file paths
file_path1 = '/Users/dgaio/github/metadata_mining/source_data/sample.coordinates.reparsed.filtered'
file_path2 = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize400_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240104_1539.txt'

###########################
# parse christain's file: 
import pandas as pd

def parse_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into parts
            parts = line.strip().split()

            # Check if the line has the expected number of parts
            if len(parts) == 5 and parts[0] == 'OUTPUT:' and parts[1] == 'sample':
                id_ = parts[2]  # Extract ID
                coords = parts[3] + ' ' + parts[4]  # Concatenate coordinates
                data.append([id_, coords])
    
    return pd.DataFrame(data, columns=['sample', 'coordinates'])


# Parse the file into a DataFrame
df1 = parse_file(file_path1)

# Display the first few rows of the DataFrame
print(df1.head())
###########################

###########################
# parse my file: 
    

file_path2 = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize400_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240104_1539.txt'

# Initialize an empty list to store the data
data = []

# Read the file
with open(file_path2, 'r') as file:
    next(file)  # Skip the header line
    for line in file:
        parts = line.strip().split(',', 1)  # Split into two parts at the first comma
        if len(parts) == 2:
            sample = parts[0].strip().replace('"', '')  # Remove quotes and strip whitespace
            coordinates = parts[1].replace(',', ' ').strip().replace('"', '')  # Replace commas, remove quotes, strip whitespace

            # Handle 'NA' and 'NA NA' values, replace with NaN
            if coordinates in ['NA NA', 'NA']:
                coordinates = np.nan

            data.append([sample, coordinates])



# Create a DataFrame from the data
df2 = pd.DataFrame(data, columns=['sample', 'coordinates'])

# Display the first few rows of the DataFrame
print(df2.head())
###########################



#Merge the two DataFrames
merged_df = pd.merge(df1, df2, left_on='sample', right_on='sample', how='right')

# Display the first few rows of the merged DataFrame
print(merged_df.head())



# parse my df: 
# remove quotes when present : """69 08 N 20 82 E"""
# if digit followed by S or W, make it negative
# 





