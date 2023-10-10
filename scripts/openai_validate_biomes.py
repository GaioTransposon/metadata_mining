#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:25:32 2023

@author: dgaio
"""


import os
import openai
import pandas as pd
import numpy as np  
from collections import Counter
import argparse  
import pickle
import re
from datetime import datetime




####################

parser = argparse.ArgumentParser(description='Process XML files.')

parser.add_argument('--work_dir', type=str, required=True, help='path to work directory')
parser.add_argument('--input_gold_dict', type=str, required=True, help='path to input df')
parser.add_argument('--input_df', type=str, required=True, help='path to input df')
parser.add_argument('--directory_with_split_metadata', type=str, required=True)
parser.add_argument('--output_plot', type=str, required=True, help='name of output plot')

args = parser.parse_args()

# Prepend work_dir to all the file paths
input_gold_dict = os.path.join(args.work_dir, args.input_gold_dict)
input_df = os.path.join(args.work_dir, args.input_df)
directory_with_split_metadata = os.path.join(args.work_dir, args.directory_with_split_metadata)
output_plot = os.path.join(args.work_dir, args.output_plot)


# for testing purposes
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
input_df = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")
directory_with_split_metadata = os.path.join(work_dir, "sample.info_split_dirs")
output_plot = os.path.join(work_dir, "gpt_vs_curated_biomes.pdf")
# ###################


########### # 1. api key
file_path = "/Users/dgaio/my_api_key"  
try:
    with open(file_path, "r") as file:
        openai.api_key = file.read().strip()
        print(openai.api_key)  
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except IOError:
    print(f"Error reading file '{file_path}'.")
###########
    
########### # 2. open gold_dict and transform to a df
with open(input_gold_dict, 'rb') as file:
    input_gold_dict = pickle.load(file)

input_gold_dict = input_gold_dict[0] # this is because the second item is the list of pmids I processed - it was necessary to run confirm_biome_game.py, now not necessary anymore 
gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)
###########

########### # 2. fetch metadata (based on gold_dict_df), clean and save

# Pick n random samples from each category
seed = 42  # Set your seed value
random_samples = gold_dict_df.groupby('curated_biome').apply(lambda x: x.sample(n=40, random_state=seed)).reset_index(drop=True)

# for the moment remove the unknown: 
random_samples = random_samples[random_samples['curated_biome'] != 'unknown']
print(random_samples)






# 1. Utility to fetch metadata from folders
def fetch_metadata_from_sample(sample, directory_with_split_metadata):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(directory_with_split_metadata, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}.txt")
    with open(metadata_file_path, 'r') as f:
        metadata = f.read()
    return metadata

# 2. loop through: 
seed = 42
shuffled_samples = random_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
len(shuffled_samples)

# endings to filter out ("$" makes sure it will skip these if preceeded by a character or empty space)
endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]

# loop through 
metadata_dict = {}
for _, row in shuffled_samples.iterrows():
    metadata = fetch_metadata_from_sample(row['sample'], directory_with_split_metadata)
    
    
    print("Metadata for", row['sample'])
    
    cleaned_metadata_lines = []
    for line in metadata.splitlines():
        stripped_line = line.strip()  # strip whitespace
        should_keep = True

        if stripped_line.lower().startswith(("experiment", "run", ">")):
            should_keep = False
        else:
            for ending in endings_to_remove:
                if re.search(ending, stripped_line, re.IGNORECASE):
                    print(f"Rejected line (ends with {ending}): {stripped_line}")
                    should_keep = False
                    break

        if should_keep:
            cleaned_metadata_lines.append(stripped_line)
    
    cleaned_metadata = "\n".join(cleaned_metadata_lines)

    # Save to dictionary
    metadata_dict[row['sample']] = cleaned_metadata

    print("Cleaned metadata:")
    print(cleaned_metadata)
    print("===================================")



###########

########### # 3. create chunks 

# dict to list
dict_as_list = [f"\n\n'sample_ID={key}': '{value}'" for key, value in metadata_dict.items()]

# chunking function
def split_list_by_tokens(input_list, max_tokens):
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for item in input_list:
        item_tokens = len(item.split())
        
        if current_token_count + item_tokens <= max_tokens:
            current_chunk.append(item)
            current_token_count += item_tokens
        else:
            chunks.append(current_chunk)
            current_chunk = [item]
            current_token_count = item_tokens
    
    if current_chunk:  # Add the remaining items in the last chunk
        chunks.append(current_chunk)
    
    return chunks

chunks = split_list_by_tokens(dict_as_list, 1000)
len(chunks)

max_length_chunk = max(chunks, key=len)
max_length = len(max_length_chunk)
print(f"The maximum number of items in a chunk is: {max_length}")

# Empty list to store content_strings for each chunk
content_strings = []

# Lists to store chunk details
chunk_tokens = []

# Joining the content within each chunk
for i, chunk in enumerate(chunks, 1):
    # Compute the number of tokens in the chunk
    total_tokens = sum(len(item.split()) for item in chunk)
    chunk_tokens.append(total_tokens)
    
    print(f"Chunk {i} Content (Number of items: {len(chunk)} | Total Tokens: {total_tokens}):")
    content_string = "\n".join(chunk)
    content_strings.append(content_string)  # Store the content_string
    print(f"Chunk {i} Content:")
    print(content_string)
    print("----")

###########

########### # 4. feed to gpt 

# Iterate over content_strings for openai calls
gpt_responses = []
n=0

for content_string in content_strings:
    n+=1
    print("Sending request number: ", n, " of ", len(chunks), " requests")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "Based on the metadata texts below, you have to guess where the sample each metadata text is based on, come from. Your choices are are: 'animal' (includes human), 'plant', 'water', 'soil'. Report the sample ID each time and the answer (strictly 1-word answer for each sample ID)."
            },
            {
                "role": "user",
                "content": content_string
            }
        ],
        temperature=0.1,
        max_tokens=4096,   # just to keep in mind: max allowed is 4096
        top_p=0.75,
        frequency_penalty=0,
        presence_penalty=0
    )
    gpt_responses.append(response)
    
    
# 20231005: first request at 17:53 for 8 samples 2 chunks = requests)
# 20231005: second request at 18:18 for 160 samples (30 chunks = requests)
# 20231009: third request at 11:33 for 160 samples (28 chunks = requests)
# 20231009_2: third request at 11:33 for 160 samples (30 chunks = requests)

# now with seed=42 in the shuffling: 
# 20231009_3: fourth request at 12:20 for 160 samples (28 chunks = requests)
# 20231009_4: fifth request at 12:42 for 160 samples (28 chunks = requests)
# 20231009_5: sixth request at 12:47 for 160 samples (28 chunks = requests)
# 20231009_6: seventh request at 13:26 for 160 samples (28 chunks = requests)


###########


########### # 5. extract gpt output

extracted_contents_gpt_responses = []

for response in gpt_responses:
    content = response.choices[0].message['content'].strip().splitlines()
    print(content)
    extracted_contents_gpt_responses.extend(content)
    
    
parsed_data = []

for line in extracted_contents_gpt_responses:
    # Match patterns like 'sample_ID=SRS5253951: soil' or 'Sample ID: SRS3992785 Answer: water'
    match = re.search(r'(sample[_\s]*ID|SRS|ERS|DRS)\s*[:=]\s*([\w\d]+)[\s\W]*(soil|water|plant|animal|human)', line, re.IGNORECASE)
    if match:
        sample_id = match.group(2)
        biome = match.group(3).lower()
        if biome == 'human':
            biome = 'animal'  # replace human with animal
        parsed_data.append((sample_id, biome))

len(parsed_data)

# Store the parsed data to a file

# Get the current date and time in the format YYYYMMDD_HHMMSS
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create the file name with the current date and time appended
file_name = f"/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gtp_35_output_{current_datetime}.txt"

# save: 
with open(file_name, "w") as f:
    for item in parsed_data:
        print(item)
        f.write(f"{item[0]}: {item[1]}\n")

    









































def fetch_metadata_from_sample(sample, directory_with_split_metadata):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(directory_with_split_metadata, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}.txt")
    with open(metadata_file_path, 'r') as f:
        # Filter out lines that start with "experiment", "run", or are empty
        metadata = "\n".join([line.strip() for line in f.readlines() if not line.startswith(("experiment", "run")) and line.strip() != ""])
    return metadata


# Filter rows where curated_biome doesn't match gpt_generated_biome
mismatched_rows = m[m['curated_biome'] != m['gpt_generated_biome']].sort_values(by='curated_biome')

mismatched_rows = mismatched_rows[mismatched_rows['curated_biome'] == 'plant']

# Then proceed with your operations on the filtered dataframe

# Fetch and print metadata for each mismatched sample
for index, row in mismatched_rows.iterrows():
    metadata_for_sample = fetch_metadata_from_sample(row['sample'], directory_with_split_metadata)
    print(f"Sample: {row['sample']}")
    print(f"Curated Biome: {row['curated_biome']}")
    print(f"GPT Generated Biome: {row['gpt_generated_biome']}")
    print("Metadata:")
    print(metadata_for_sample)
    print("="*40)  # Separating line for readability



