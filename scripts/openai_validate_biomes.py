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


# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
# input_df = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")
# directory_with_split_metadata = os.path.join(work_dir, "sample.info_split_dirs")
# output_plot = os.path.join(work_dir, "gpt_vs_curated_biomes.pdf")
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
shuffled_samples = random_samples.sample(frac=1).reset_index(drop=True)

# endings to filter out
endings_to_remove = ["=", "nan", "not applicable", "missing"]

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
                if stripped_line.lower().endswith(ending):
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
        temperature=1.00,
        max_tokens=4096,   # just to keep in mind: max allowed is 4096
        top_p=0.75,
        frequency_penalty=0,
        presence_penalty=0
    )
    gpt_responses.append(response)
    
    
# 20231005: first request at 17:53 for 8 samples 2 chunks = requests)
# 20231005: second request at 18:18 for 160 samples (30 chunks = requests)
###########



########### # 5. extract gpt output

extracted_contents_gpt_responses = []

for response in gpt_responses:
    content = response.choices[0].message['content'].strip().splitlines()
    print(content)
    extracted_contents_gpt_responses.extend(content)


key_terms = ['animal', 'plant', 'water', 'soil', 'unknown']


# clean the output. because it does happen that gpt includes the sample name in its answer...
def extract_samples_from_response(response):
    # Define regex patterns
    patterns = [
        r'(\w+): (\w+)',  # For "SRS699624: water"
        r'Sample ID (\w+): \'(\w+)\'',  # For "Sample ID ERS2553522: 'plant'"
        r'sample_ID=(\w+): (\w+)',  # For "sample_ID=SRS2175714: soil"
    ]

    sample_info = {}
    biome_counts = {term: 0 for term in key_terms}

    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            sample_id, biome = match
            # In case there are biomes like 'animal (fish)', extract only the first word.
            biome = biome.split()[0]
            sample_info[sample_id] = biome
            if biome in key_terms:
                biome_counts[biome] += 1
    
    # For the format "Sample ID: SRS4701369" & "Answer: soil"
    samples = re.findall(r'Sample ID: (\w+)', response)
    answers = re.findall(r'Answer: (\w+)', response)
    if len(samples) == len(answers):
        for sample, answer in zip(samples, answers):
            sample_info[sample] = answer
            if answer in key_terms:
                biome_counts[answer] += 1

    return sample_info, biome_counts

data = {}
total_biome_counts = {term: 0 for term in key_terms}
for response in extracted_contents_gpt_responses:
    extracted_data, biome_counts = extract_samples_from_response(response)
    data.update(extracted_data)
    for biome, count in biome_counts.items():
        total_biome_counts[biome] += count

print(data)
print(total_biome_counts)

# Convert the dictionary into a DataFrame
cleaned_responses_df = pd.DataFrame(list(data.items()), columns=['sample', 'gpt_generated_biome'])

print(cleaned_responses_df)

###########







########### # 6. plot output: comparing curated_biome vs gpt_generated_biome





# open df [and merge gold_dict to compare previous biomes to the manually curated ones (this will be moved to confirm_biome_game)]
input_df = pd.read_csv(input_df)
input_df.columns
input_df['biome'] = input_df['biome'].replace('aquatic', 'water')
input_df = input_df[['sample', 'biome']]


gold_dict_input_df = pd.merge(gold_dict_df, input_df, on='sample', how='inner')
gold_dict_input_df.columns

m = pd.merge(cleaned_responses_df, gold_dict_input_df, on='sample', how='inner') 
m.columns
m

len(m)
m = m.drop_duplicates()
len(m)






###########


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# Identify unique labels from both columns
labels = sorted(list(set(m['curated_biome']).union(set(m['gpt_generated_biome']))))

# Generate the confusion matrix with explicit label ordering
matrix = confusion_matrix(m['curated_biome'], m['gpt_generated_biome'], labels=labels)

# Normalize the matrix to get percentages
normalized_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] * 100

# Create combined annotations with percentages and raw counts
annotations = [["{0:.2f}%\n(n={1})".format(normalized_matrix[i, j], matrix[i, j]) 
                for j in range(len(matrix[i]))] for i in range(len(matrix))]

# Plot the heatmap
plt.figure(figsize=(10,7))
sns.heatmap(matrix, annot=annotations, fmt='', cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels)
plt.xlabel('Curated biome')
plt.ylabel('GPT-generated biome')
plt.title('Confusion Matrix for GPT Predictions vs Curated Biomes')
plt.show()






import matplotlib.pyplot as plt

# Step 1: Filter dataframe and sort
m_filtered = m[['sample', 'biome', 'curated_biome', 'gpt_generated_biome']]
m_filtered = m_filtered.sort_values(by='curated_biome')

# Step 2: Create color function
def assign_color(value):
    color_dict = {
        'animal': 'pink',
        'water': 'blue',
        'soil': 'yellow',
        'plant': 'green',
        'unknown': 'lightgray'
    }
    return color_dict.get(value, 'white')  # Default to white if no match


def plot_table(data):
    colors = [[assign_color(cell) for cell in row] for _, row in data.iterrows()]

    fig, ax = plt.subplots(figsize=(12, len(data)*0.5))
    ax.axis('off')
    
    tbl = ax.table(cellText=data.values, cellColours=colors, 
                   colLabels=data.columns, cellLoc='center', loc='center')
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(data.columns))))
    plt.show()

# Breaking the data into chunks of, for instance, 20 rows per figure
n = 25  # Number of rows per chunk
chunks = [m_filtered.iloc[i:i + n] for i in range(0, len(m_filtered), n)]

for chunk in chunks:
    plot_table(chunk)





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



