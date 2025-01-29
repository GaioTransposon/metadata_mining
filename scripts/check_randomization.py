# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""




import os
import pickle
import pandas as pd
import re

# Load data from pkl file
pkl_file = '/Users/danielagaio/github/metadata_mining/source_data/gold_dict.pkl'  # Replace with your actual pkl file path
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Extract first values from pkl file into gold_dict
gold_dict = {key: value[1] for key, value in data.items()}

# Directory where the metadata_chunks_* files are stored
directory = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/'  # Replace with your actual directory path

# Dictionary to store DataFrames
dfs = {}

# Process each file in the directory
for filename in os.listdir(directory):
    if filename.startswith('metadata_chunks_'):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        
        # Parse the txt file
        biome_data = []
        with open(file_path, 'r') as f:
            content = f.read()  # Read the entire file content

            # Use a regex pattern to capture sample IDs from the text
            sample_pattern = r"'sample_ID=(\S+)': '>(\S+)\n.*?-----"
            matches = re.findall(sample_pattern, content, re.DOTALL)

            # Populate the biome_data list with sample_id and corresponding biome from gold_dict
            for match in matches:
                sample_id, _ = match  # We don't need the biome value from the text file, just the sample ID
                biome = gold_dict.get(sample_id, 'Unknown')  # Use 'Unknown' if sample_id is not in gold_dict
                biome_data.append({'sample_id': sample_id, 'biome': biome})

        # Create a DataFrame from the parsed data
        biome_df = pd.DataFrame(biome_data)

        # Only keep DataFrames that have at least 10 rows
        if len(biome_df) > 50:
            # Save the DataFrame to the dictionary, with the filename (without 'metadata_' prefix) as the key
            df_name = filename.replace('metadata_', '').replace('.txt', '')
            dfs[df_name] = biome_df


import pandas as pd

# Function to check if biomes are likely randomized
def check_randomized_biomes(df):
    # Calculate the number of transitions (changes in biome) between consecutive rows
    transitions = (df['biome'] != df['biome'].shift()).sum()
    
    # Compare the number of transitions to the length of the dataframe
    # More transitions suggest randomization, fewer transitions suggest clustering (non-random)
    transition_ratio = transitions / len(df)
    
    # Heuristic: If transitions are more than 10% of the total, it's likely randomized
    if transition_ratio > 0.1:  
        return True  # Likely randomized
    else:
        return False  # Likely not randomized

# Dictionary to store results
randomized_status = {}

# Check each DataFrame in 'dfs'
for df_name, df in dfs.items():
    is_randomized = check_randomized_biomes(df)
    randomized_status[df_name] = 'Randomized' if is_randomized else 'Not Randomized'

# Display results
for df_name, status in randomized_status.items():
    print(f"{df_name}: {status}")




# Function to check if biomes are likely randomized
def check_randomized_biomes(df):
    repeated_biomes = set()
    seen_biomes = set()

    for i in range(1, len(df)):
        current_biome = df.iloc[i]['biome']
        previous_biome = df.iloc[i - 1]['biome']
        
        # Check if the current biome has appeared before and is not in consecutive rows
        if current_biome in seen_biomes and current_biome != previous_biome:
            repeated_biomes.add(current_biome)
        
        # Add biome to the seen set (for checking future occurrences)
        seen_biomes.add(previous_biome)

    # If any biome has appeared non-consecutively, we classify it as randomized
    if len(repeated_biomes) > 0:
        return True  # Randomized
    else:
        return False  # Not randomized

# Dictionary to store results
randomized_status = {}

# Check each DataFrame in 'dfs'
for df_name, df in dfs.items():
    is_randomized = check_randomized_biomes(df)
    randomized_status[df_name] = 'Randomized' if is_randomized else 'Not Randomized'

# Sort the results by the file name (keys)
sorted_randomized_status = {k: randomized_status[k] for k in sorted(randomized_status)}

# Display sorted results
for df_name, status in sorted_randomized_status.items():
    print(f"{df_name}: {status}")






