#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:25:32 2023

@author: dgaio
"""


########### # 1. Environment Setup Script: setup_environment.py

import os
import openai
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import pickle
import re
from datetime import datetime


########### # 2. Arguments Parser Script: parse_args.py

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='path to work directory')
    parser.add_argument('--input_gold_dict', type=str, required=True, help='name of ground truth dictionary')
    parser.add_argument('--input_df', type=str, required=True, help='suggested: sample.info_biome_pmid_title_abstract.csv')
    parser.add_argument('--directory_with_split_metadata', type=str, required=True)

    return parser.parse_args()

args = parse_arguments()

# Prepend work_dir to all the file paths
input_gold_dict = os.path.join(args.work_dir, args.input_gold_dict)
input_df = os.path.join(args.work_dir, args.input_df)
directory_with_split_metadata = os.path.join(args.work_dir, args.directory_with_split_metadata)


# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
# input_df = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")
# directory_with_split_metadata = os.path.join(work_dir, "sample.info_split_dirs")
# ###################


########### # 3. Metadata Processing Script: process_metadata.py

def load_gold_dict(input_gold_dict_path):
    with open(input_gold_dict_path, 'rb') as file:
        input_gold_dict = pickle.load(file)
        return input_gold_dict[0] # this is because the second item is the list of pmids I processed - it was necessary to run confirm_biome_game.py, now not necessary anymore 

def transform_gold_dict_to_df(input_gold_dict):
    gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
    gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
    gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
    gold_dict_df.drop(columns='tuple_data', inplace=True)
    return gold_dict_df

def get_random_samples(gold_dict_df, n=40, seed=42):
    random_samples = gold_dict_df.groupby('curated_biome').apply(lambda x: x.sample(n=n, random_state=seed)).reset_index(drop=True)
    # for the moment remove the unknown: 
    return random_samples[random_samples['curated_biome'] != 'unknown'] 

def fetch_metadata_from_sample(sample, directory_with_split_metadata):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(directory_with_split_metadata, folder_name)  
    metadata_file_path = os.path.join(folder_path, f"{sample}.txt")
    with open(metadata_file_path, 'r') as f:
        return f.read()

def process_metadata(samples, directory_with_split_metadata):
    seed = 42
    shuffled_samples = samples.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # endings to filter out ("$" makes sure it will skip these if preceeded by a character or empty space)
    endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
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
        metadata_dict[row['sample']] = cleaned_metadata
        
        print("Cleaned metadata:")
        print(cleaned_metadata)
        print("===================================")
        
    return metadata_dict

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
    if current_chunk:
        chunks.append(current_chunk)
    
    print('Number of chunks: ',len(chunks))
    
    max_length_chunk = max(chunks, key=len)
    max_length = len(max_length_chunk)
    print(f"The maximum number of items in a chunk is: {max_length}")
    
    return chunks

def create_chunks(metadata_dict, max_tokens=1000):
    dict_as_list = [f"\n\n'sample_ID={key}': '{value}'" for key, value in metadata_dict.items()]
    return split_list_by_tokens(dict_as_list, max_tokens)



# in --> -
gold_dict = load_gold_dict(input_gold_dict)

# middle --> -
gold_dict_df = transform_gold_dict_to_df(gold_dict)

# middle --> -
random_samples = get_random_samples(gold_dict_df)

# in + middle --> -
metadata_dict = process_metadata(random_samples, directory_with_split_metadata)

# middle --> out
chunks = create_chunks(metadata_dict)



########### # 4. GPT Interaction Script: interact_with_gpt.py


def consolidate_chunks_to_strings(chunks):
    """
    Consolidate individual items within each chunk into one content string.

    Parameters:
    - chunks: List of lists containing the metadata chunked by tokens.

    Returns:
    - List of consolidated content strings from each chunk.
    """
    # Empty list to store content_strings for each chunk
    content_strings = []

    # Lists to store chunk details (can be returned if needed)
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
    
    return content_strings  # Return the consolidated content_strings


def load_api_key(file_path):
    """
    Load OpenAI API key from a file.

    Parameters:
    - file_path: Path to the file containing the OpenAI API key.

    Returns:
    - The OpenAI API key as a string.
    """
    try:
        with open(file_path, "r") as file:
            api_key = file.read().strip()
            return api_key  
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except IOError:
        print(f"Error reading file '{file_path}'.")
        return None

def gpt_request(content_string, my_model, my_temperature):
    """
    Make a request to GPT for metadata processing.

    Parameters:
    - content_string: String content to process by GPT.

    Returns:
    - Response from GPT.
    """
    return openai.ChatCompletion.create(
        model=my_model,
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
        temperature=my_temperature,
        max_tokens=4096,   # just to keep in mind: max allowed is 4096
        top_p=0.75,
        frequency_penalty=0,
        presence_penalty=0
    )

def interact_with_gpt(content_strings, my_model, my_temperature):
    """
    Iterate over content_strings and make requests to GPT.

    Parameters:
    - content_strings: List of string contents to process.

    Returns:
    - List of responses from GPT-3.
    """
    gpt_responses = []
    for index, content_string in enumerate(content_strings, start=1):
        print(f"Sending request number: {index} of {len(content_strings)} requests")
        response = gpt_request(content_string=content_string, my_model=my_model, my_temperature=my_temperature)
        gpt_responses.append(response)
    return gpt_responses


def save_gpt_responses_to_file(work_dir, gpt_responses, my_model, my_temperature):
    """
    Save the content of GPT responses to a file.

    Parameters:
    - gpt_responses: List of GPT responses.
    - model: Name of the GPT model.
    - temperature: The temperature setting used during the request.

    Returns:
    - None
    """
    # Extract the "content" from each response
    contents = [response['choices'][0]['message']['content'] for response in gpt_responses]
    #print(contents)
    # Join all contents with a separator (two newlines for readability)
    final_content = "\n\n".join(contents)
    print(final_content)
    
    # Construct the filename
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"gtp_output_model{my_model}_temp{my_temperature}_dt{current_datetime}.txt"
    filename = os.path.join(work_dir, filename)
    
    # Write to the file
    with open(filename, 'w') as file:
        file.write(final_content)

    print(f"Saved GPT responses to: {filename}")
    


   

# in
content_strings = consolidate_chunks_to_strings(chunks)

# in
file_path = "/Users/dgaio/my_api_key"
openai.api_key = load_api_key(file_path)



my_model = "gpt-3.5-turbo-16k-0613"
my_temperature = 1.00

# middle --> out 
if openai.api_key:
    print("openai api key succesfully read")
    responses = interact_with_gpt(content_strings, my_model=my_model, my_temperature=my_temperature)
    

   
save_gpt_responses_to_file(work_dir, responses, my_model, my_temperature)



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






















with open("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gtp_output_modelgpt-3.5-turbo-16k-0613_temp1.0_dt20231011_175649.txt", 'r') as file:
    lines = file.readlines()

# how many can we expect? 
keywords = ['animal', 'soil', 'water', 'plant', 'human']
total_count = 0

for line in lines:
    for keyword in keywords:
        total_count += line.lower().count(keyword)

print(f"Total count of mentions: {total_count}")





import re

def parse_samples(lines):
    result = {}
    sample_id_pattern = re.compile(r'(SRS|ERS|DRS)\d+')
    buffer = None  # To store the ID temporarily

    for line in lines:
        id_match = sample_id_pattern.search(line)

        if id_match:
            if buffer:  # If there's a previous ID buffered, store it with its text
                result[buffer] = text.strip()
            buffer = id_match.group()  # Store the new ID
            text = line[id_match.end():]  # Start storing text from after the ID match
        elif buffer:
            text += ' ' + line  # If there's an ID buffered, continue adding text

    # Add the last buffered item
    if buffer:
        result[buffer] = text.strip()

    return result

parsed_data = parse_samples(lines)

# Printing the parsed data
for k, v in parsed_data.items():
    print(f"Sample ID: {k}, Text: {v}")
len(parsed_data)




# to do: 
# n=1 still not picked up. solve. 
# store into df. column 'gpt_generated_output_raw' column 'gpt_generated_output_clean'









extracted_contents_gpt_responses = []

for response in gpt_responses:
    content = response.choices[0].message['content']#.strip()#.splitlines()
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



