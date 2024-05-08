#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:36:00 2024

@author: dgaio
"""


import json
from openai import OpenAI
import pandas as pd
from IPython.display import Image, display
import os



# Initializing OpenAI client
api_key_file = os.path.expanduser("~/my_api_key")
with open(api_key_file, "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)


# =============================================================================
# '''We kindly request your expertise in analyzing the following microbial metagenomic samples from their metadata texts:
# - Please deduce the source category for the sample, choosing from 'animal' (including humans), 'plant', 'water', 'soil', or 'other'. Your choices are: 'animal' (incl. human), 'plant', 'water', 'soil', 'other'. Give strictly a concise, 
# single-word label for the sample ID.
# - Please infer geographical location where the sample was collected, including the country (NOT the coordinates).
# -  Extract strictly 5 to 8 keywords descriptive of the sample origin, separated by commas. Put them within curly brackets.
# -  We seek a brief, up to three-word description of the sample's specific origin. For 'animal' or 'plant' sources, please specify the host and part thereof. For 'water' samples, the type of water body is sought (e.g., 
# lake, brine, sea, waste water, etc). If from 'soil', specify (e.g.: agricultural, desert, forest, etc). If from 'other' specify which (e.g.: urban, laboratory, feed/food, fungus, air, etc).
# 
# If information is missing, kindly indicate 'NA'. Please separate all values with three underscores ('___').
# An example response: SRS123456___animal___Los Angeles, USA___{medical, bone fracture, infection, collagen, hospital, intensive care, cast, Staphilococcus epidermidis}___human elbow'''
# 
# =============================================================================




# transform metadata_chunk.txt file to csv: 
import csv
import pickle

# Define the input and output file paths
input_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202405061728.txt"
input_pkl_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_prov.pkl"
output_csv_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202405061728_nolines.csv"



# Function to clean text by replacing unwanted characters and newlines
def clean_text(text):
    return text.replace("'", "").replace('"', "").replace("\n", "\\n")

# Load the pickle file
with open(input_pkl_path, 'rb') as pkl_file:
    data_dict = pickle.load(pkl_file)

# Write the cleaned data to a CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['sample_id', 'metadata'])  # Write the header row
    
    # Iterate over each item in the dictionary and write to the CSV file
    for sample_id, text in data_dict.items():
        cleaned_text = clean_text(text)  # Clean the text
        writer.writerow([sample_id, cleaned_text])  # Write to the CSV

print("Data has been processed and written to CSV.")




dataset_path = output_csv_path #"/Users/dgaio/Downloads/imbd_microbeatlas3.csv"

df = pd.read_csv(dataset_path)
df.head()


categorize_system_prompt = '''
We kindly request your expertise in analyzing the microbial metagenomic samples from their metadata texts.
Please deduce the source category for the sample, choosing from 'animal' (including humans), 'plant', 'water', 'soil', or 'other'. Your choices are: 'animal' (incl. human), 'plant', 'water', 'soil', 'other'. Give strictly a concise, 
single-word label for the sample ID.
You will be provided with the metadata texts, and you will output a json object containing the following information:

{
    sample_id: string // the sample id
    biome_label: string // the biological origin of the sample
    geo_location: string[] // geographical location where the sample was collected, including the country (NOT the coordinates)
}

For the biome_label choose from 'animal' (including human), 'plant', 'water', 'soil', or 'other'. Give strictly a single-word biome label for each sample ID. 
If information is missing, kindly indicate 'NA'.
'''



def get_categories(description, sample_id):
    full_description = f"Sample ID: {sample_id}, Metadata: {description}"  # Formatting the string with sample_id and metadata
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": categorize_system_prompt},
            {"role": "user", "content": full_description}
        ],
    )
    biome_label = response.choices[0].message.content
    return {"sample_id": sample_id, "biome_label": biome_label}

for _, row in df[:5].iterrows():
    metadata = row['metadata']
    sample_id = row['sample_id']
    result = get_categories(metadata, sample_id)
    print(f"Sample ID: {result['sample_id']}\nMetadata: {metadata}\n\nRESULT: {result['biome_label']}")
    print("\n\n----------------------------\n\n")




# Creating an array of json tasks
tasks = []

for index, row in df.iterrows():
    
    metadata = row['metadata']
    sample_id = row['sample_id'] 
    
    task = {
        "custom_id": f"task-{sample_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": metadata
                }
            ],
        }
    }
    
    tasks.append(task)
    
print(tasks)


# Creating an array of json tasks
tasks = []

for index, row in df.iterrows():
    metadata = row['metadata']
    sample_id = row['sample_id']
    
    # Include the sample ID in the metadata string
    user_content = f"Sample ID: {sample_id}, Metadata: {metadata}"  # This concatenates sample ID with metadata
    
    task = {
        "custom_id": f"task-{sample_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": user_content  # Updated to use the combined string
                }
            ],
        }
    }
    
    tasks.append(task)

print(tasks)

    
    
# Creating the file

file_name = "/Users/dgaio/Downloads/batch_tasks_metadata.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        print(obj)
        file.write(json.dumps(obj) + '\n')
    

batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)
print(batch_file)



# Creating the batch job
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)


# movies: batch_mgceyD6JG1QRRMtV84snBap4
# metadata: batch_BZmlZyQBzm0OpnelNaXEhlRs
# metadata with sample ids in task: batch_fzN04mDCeGre54RmssyZmkBH
# batch_BZmlZyQBzm0OpnelNaXEhlRs
# hopefully with sample id s included: batch_UnGVBX6rFxHS3Dv3YzgANIp3

# Checking batch status
batch_job = client.batches.retrieve('batch_UnGVBX6rFxHS3Dv3YzgANIp3')  #batch_job.id
print(batch_job)




# Retrieving results 
result_file_id = batch_job.output_file_id
print(result_file_id)

result = client.files.content(result_file_id).content
result_file_name = "/Users/dgaio/Downloads/batch_job_results_metadata.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)

# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
        
print(results)




# convert json results to csv: 
import json
import csv

# Define the input JSONL file and output CSV file paths
input_jsonl_path = result_file_name
output_csv_path = "/Users/dgaio/Downloads/batch_job_results_metadata.csv"

# Open the JSONL file and the CSV file for writing
with open(input_jsonl_path, 'r') as jsonl_file, open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the CSV header
    writer.writerow(['col_0', 'col_1', 'col_2'])
    
    # Read each line from the JSONL file
    for line in jsonl_file:
        json_obj = json.loads(line)  # Parse the line as JSON
        # Navigate to the nested 'content' field
        content_str = json_obj['response']['body']['choices'][0]['message']['content']
        print(content_str)
        
        # Parse the JSON string from 'content'
        content_data = json.loads(content_str)
        print(content_data)
        
        # Extract the desired fields
        sample_id = content_data['sample_id'] # Assuming sample_id is always a list with at least one item
        biome_label = content_data['biome_label']
        geo_location = content_data['geo_location'][0]  # Assuming geo_location is always a list with at least one item
        
        # Write to the CSV file
        writer.writerow([sample_id, biome_label, geo_location])

print("Data has been processed and written to CSV.")




























