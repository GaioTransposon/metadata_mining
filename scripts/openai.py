#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:04:57 2023

@author: dgaio
"""


import os
import openai

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


response = openai.Completion.create(
  model="gpt-3.5-turbo",
  prompt="Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\nTweet: \"I loved the new Batman movie!\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)

print(response)


import openai
import pandas as pd

# Group by sample, then convert each group's pmids to a frozenset (to make it hashable)
grouped_by_sample = filtered_df.groupby('sample')['title'].apply(frozenset)
len(grouped_by_sample)

# Filter out samples with only 1 pmid
grouped_by_sample = grouped_by_sample[grouped_by_sample.apply(len) > 1]
len(grouped_by_sample)

# Now, keep only the unique sets of pmids
unique_groups = grouped_by_sample.unique()
len(unique_groups)

# Convert back to list for further processing
unique_groups_list = [list(group) for group in unique_groups]
len(unique_groups_list)




# Function to ask GPT-3 which title does not belong
def outlier_title(titles):
    joined_titles = ', '.join([f"title {i+1}: {title}" for i, title in enumerate(titles)])
    prompt = f"Decide which title does not belong with the majority: {joined_titles}"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )

    return response.choices[0].text.strip()

# Get the outlier for each group of pmids
outliers = {}
for pmids in grouped_pmids:
    outlier = outlier_title(pmids)
    outliers[pmids] = outlier

print(outliers)


