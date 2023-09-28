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

# get key 
file_path = "/Users/dgaio/my_api_key"  # Replace with the actual file path
try:
    with open(file_path, "r") as file:
        openai.api_key = file.read().strip()
        print(openai.api_key)  
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except IOError:
    print(f"Error reading file '{file_path}'.")









# 1. Filter out samples with just 1 pmid
filtered_df2 = filtered_df.groupby('sample').filter(lambda x: len(x) > 1)


# 2. Keep only 'pmid' and 'title' columns
filtered_df2 = filtered_df2[['pmid', 'title']]

# 3. Keep only unique rows
filtered_df2 = filtered_df2.drop_duplicates()


len(filtered_df2)




z = filtered_df2



# Create the desired list
result = [f"'{pmid}': '{title}'" for pmid, title in zip(z['pmid'], z['title'])]
#print(result)

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

chunks = split_list_by_tokens(result, 2500)
len(chunks)

# Find the chunk with the maximum number of items
max_length_chunk = max(chunks, key=len)

# Get the length of this chunk
max_length = len(max_length_chunk)

print(f"The maximum number of items in a chunk is: {max_length}")




# Empty list to store content_strings for each chunk
content_strings = []

# Joining the content within each chunk
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} Content (Number of items: {len(chunk)}):")
    content_string = "\n".join(chunk)
    content_strings.append(content_string)  # Store the content_string
    print(f"Chunk {i} Content:")
    print(content_string)
    print("----")





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




# run at 17:56

# Iterate over content_strings for openai calls
responses = []
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
    responses.append(response)
    
    
    
responses[2]    




from collections import Counter

for response in responses:
    content = response.choices[0].message['content'].strip().splitlines()
    
    counts = {
        'animal': sum(1 for line in content if 'animal' in line.lower()),
        'plant': sum(1 for line in content if 'plant' in line.lower()),
        'water': sum(1 for line in content if 'water' in line.lower()),
        'soil': sum(1 for line in content if 'soil' in line.lower()),
        'unknown': sum(1 for line in content if 'unknown' in line.lower()),
    }

    print(f"Length of content: {len(content)}")
    print(f"Occurrences - Animal: {counts['animal']}, Plant: {counts['plant']}, Water: {counts['water']}, Soil: {counts['soil']}, Unknown: {counts['unknown']}\n")




extracted_contents = []

for response in responses:
    content = response.choices[0].message['content'].strip().splitlines()
    print(len(content))
    extracted_contents.extend(content)

# At this point, extracted_contents contains all the desired content from the openai calls
len(extracted_contents)




content_strings[0]
responses[0]















