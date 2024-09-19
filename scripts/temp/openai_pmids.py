#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:04:57 2023

@author: dgaio
"""


import os
import openai
import pandas as pd
import numpy as np  
from collections import Counter
import argparse  



####################

parser = argparse.ArgumentParser(description='Process XML files.')

parser.add_argument('--work_dir', type=str, required=True, help='path to work directory')
parser.add_argument('--input', type=str, required=True, help='path to input df')
parser.add_argument('--output_file', type=str, required=True, help='name of output file')

args = parser.parse_args()

# Prepend work_dir to all the file paths
sample_info_biome_pmid_title_abstract = os.path.join(args.work_dir, args.input)
output_file = os.path.join(args.work_dir, args.output_file)


# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# sample_info_biome_pmid_title_abstract = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")
# output_file = os.path.join(work_dir, "training_data_pmids_based.csv")
# ###################



############ 1. open input df
sample_info_biome_pmid_title_abstract = pd.read_csv(sample_info_biome_pmid_title_abstract)









# get key 
file_path = "/Users/dgaio/my_api_key"  
try:
    with open(file_path, "r") as file:
        openai.api_key = file.read().strip()
        print(openai.api_key)  
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except IOError:
    print(f"Error reading file '{file_path}'.")









# 1. Filter out samples with just 1 pmid
filtered_df2 = sample_info_biome_pmid_title_abstract.groupby('sample').filter(lambda x: len(x) > 1)


# 2. Keep only 'pmid' and 'title' columns
filtered_df2 = filtered_df2[['pmid', 'title']]

# 3. Keep only unique rows
filtered_df2 = filtered_df2.drop_duplicates()


len(filtered_df2)




z = filtered_df2



# Create the desired list
result = [f"'{pmid}': '{title}'" for pmid, title in zip(z['pmid'], z['title'])]
#print(result)
len(result)

for item in result:
    print(item)


# split the list by max number of tokens. 
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

chunks = split_list_by_tokens(result, 1000) # chunk of 2500 gave ca 2x > gpt output. input n=2345 --> output n=4621 ! unique pmids! like it's making stuff up! 
len(chunks)

# Find the chunk with the maximum number of items
max_length_chunk = max(chunks, key=len)

# Get the length of this chunk
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

# Compute statistics on the total tokens per chunk
average_tokens = np.mean(chunk_tokens)
median_tokens = np.median(chunk_tokens)
max_tokens = np.max(chunk_tokens)
min_tokens = np.min(chunk_tokens)

print("Statistics on total tokens per chunk:")
print(f"Average: {average_tokens:.2f}")
print(f"Median: {median_tokens}")
print(f"Max: {max_tokens}")
print(f"Min: {min_tokens}")



# so if max input tokens is 2498 per chunk, 
# and we have a system prompt of 44 tokens here below
# and each chunk contains a max of 178 items
# and we expect per item 6 tokens output. 
# then we need to ask for: 
max_tokens+44+(178*6)




# run at 17:56 # run again on 20230929

# Iterate over content_strings for openai calls
responses_1000_chunk = []
n=0

for content_string in content_strings:
    n+=1
    print("Sending request number: ", n, " of ", len(chunks), " requests")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "Based on the study titles below, you have to guess where the sample(s) the studies are based on, come from. Your choices are are: 'animal' (includes human), 'plant', 'water', 'soil', 'unknown'. Report the pmid each time and the answer (strictly 1-word answer for each pmid)."
            },
            {
                "role": "user",
                "content": content_string
            }
        ],
        temperature=1.03,
        max_tokens=4096,   # just to keep in mind: max allowed is 4096
        top_p=0.75,
        frequency_penalty=0,
        presence_penalty=0
    )
    responses_1000_chunk.append(response)
    
    
    
responses_1000_chunk







extracted_contents_1000_chunk = []
key_terms = ['animal', 'plant', 'water', 'soil', 'unknown']

for response in responses_1000_chunk:
    content = response.choices[0].message['content'].strip().splitlines()

    # Count the occurrences of key terms
    counts = Counter()
    for line in content:
        for term in key_terms:
            if term in line.lower():
                counts[term] += 1

    print(f"Length of content: {len(content)}")
    print(f"Occurrences - Animal: {counts['animal']}, Plant: {counts['plant']}, Water: {counts['water']}, Soil: {counts['soil']}, Unknown: {counts['unknown']}\n")

    # Extend the extracted contents list
    extracted_contents_1000_chunk.extend(content)

# At this point, extracted_contents contains all the desired content from the openai calls
print("Total extracted contents:", len(extracted_contents_1000_chunk))



# =============================================================================
# # Extract unique PMIDs
# unique_pmids_out = set()
# for item in extracted_contents_1000_chunk:
#     pmid = item.split('.')[0]
#     unique_pmids_out.add(pmid)
# 
# # Print the number of unique PMIDs and unique items in the list
# print("Number of unique PMIDs:", len(unique_pmids_out))
# print("Number of unique items in the list:", len(set(extracted_contents_1000_chunk)))
# =============================================================================






# Extract PMIDs and values from extracted_contents_1000_chunk
pmid_values = [item.split(': ') for item in extracted_contents_1000_chunk]
pmid_df = pd.DataFrame(pmid_values, columns=['pmid', 'value'])


def remove_single_quotes(df, column_name='value'):
    df_copy = df.copy()
    df_copy.loc[:, column_name] = df_copy[column_name].str.replace("'", "", regex=False)
    return df_copy


pmid_df = remove_single_quotes(pmid_df, 'value')
print(pmid_df)





# Convert the 'pmid' column to int for matching
pmid_df['pmid'] = pmid_df['pmid'].str.replace("'", "").astype(int)







sample_info_biome_pmid_title_abstract_copy = sample_info_biome_pmid_title_abstract.copy()
# Convert the 'pmid' column of filtered_df2 to int for matching
sample_info_biome_pmid_title_abstract_copy['pmid'] = sample_info_biome_pmid_title_abstract_copy['pmid'].astype(int)

# Now merge
merged_df = pd.merge(sample_info_biome_pmid_title_abstract_copy, pmid_df, on='pmid', how='left')



zz = merged_df
print(zz)



# Function to determine the majority value
def get_majority_value(group):
    count = group.value_counts()
    
    if len(count) == 0:  # if there are no non-null values in the group
        return None

    if count.index[0] == 'unknown':
        # Return the second majority vote if 'unknown' is the majority
        return count.index[1] if len(count) > 1 else count.index[0]
    return count.index[0]

# Get majority value for each group
majority_series = zz.groupby('sample')['value'].apply(get_majority_value)

zzz = zz.copy()

# Map the majority values to the original dataframe
zzz['majority'] = zzz['sample'].map(majority_series)
print(zzz)



zzzz = zzz.copy()


# Using the previous zz_copy
# 1. Remove rows where value doesn't match majority unless majority is None
zzzz = zzzz[((zzzz['value'] == zzzz['majority']) | (zzzz['majority'].isnull()))]

# 2. Select the oldest pmid per sample
zzzz = zzzz.sort_values('pmid').groupby('sample').first().reset_index()

# 3. Drop the 'value' column
zzzz.drop('value', axis=1, inplace=True)

# 4. Rename the 'majority' column to 'biome_gpt'
zzzz.rename(columns={'majority': 'biome_gpt'}, inplace=True)











samples_with_1pmid = sample_info_biome_pmid_title_abstract_copy.groupby('sample').filter(lambda x: len(x) == 1)


len(samples_with_1pmid)
len(zzzz)






# Add a missing column to the first DataFrame with the fill value
samples_with_1pmid['biome_gpt'] = 'not_enquired'

# Concatenate the two DataFrames
final = pd.concat([samples_with_1pmid, zzzz], axis=0, ignore_index=True)

print(final)
len(final)



# Save
final.to_csv(os.path.join(output_file), index=False)
print("Output file succesfully written")




# test = final[final['biome'] == 'plant']
# uu = test['pmid'].value_counts()
# print(uu)
# len(uu)


