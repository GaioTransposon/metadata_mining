#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:36:00 2024

@author: dgaio
"""


import json
import os
import logging
import pandas as pd
from openai import OpenAI
from datetime import datetime

def init_openai_client(api_key_path):
    """Initialize the OpenAI client with API key."""
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)

def load_system_prompt(work_dir, system_prompt_file):
    """Load the system prompt from a specified file."""
    prompt_file = os.path.join(work_dir, system_prompt_file)
    try:
        with open(prompt_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"System prompt file '{prompt_file}' not found.")
        return None
    except IOError:
        logging.error(f"Error reading system prompt file '{prompt_file}'.")
        return None
    
    
def get_categories(client, description, sample_id, system_prompt):
    """Categorize with gpt based on the metadata description."""
    full_description = f"Sample ID: {sample_id}, Metadata: {description}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_description}
        ],
    )
    biome_label = response.choices[0].message.content
    return {"sample_id": sample_id, "biome_label": biome_label}


def prepare_batch_tasks(df, output_file_path, system_prompt, model, temperature):
    """Prepare batch tasks and save them to a specified file."""
    tasks = []
    for _, row in df.iterrows():
        user_content = f"Sample ID: {row['sample_id']}, Metadata: {row['metadata']}"
        task = {
            "custom_id": f"task-{row['sample_id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "response_format": { "type": "json_object" },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
            }
        }
        tasks.append(task)
    with open(output_file_path, 'w') as file:
        for task in tasks:
            file.write(json.dumps(task) + '\n')
            
            
            
            
# Setup and Configuration

# Paths and client initialization
api_key_file = os.path.expanduser("~/my_api_key")
client = init_openai_client(api_key_file)

work_dir = "/Users/dgaio/github/metadata_mining/source_data"
prompt_file_name = "openai_system_prompt_batch.txt"
categorize_system_prompt = load_system_prompt(work_dir, prompt_file_name)

# Load data and execute categorization
data_file = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_prov.csv"
df = pd.read_csv(data_file)
# for _, row in df.head(5).iterrows():
#     result = get_categories(client, row['metadata'], row['sample_id'], categorize_system_prompt)
#     print(f"Sample ID: {result['sample_id']}\nMetadata: {row['metadata']}\n\nRESULT: {result['biome_label']}")
#     print("\n----------------------------\n")

# Prepare and execute batch processing
file_name = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_tasks_metadata.jsonl"
model = "gpt-3.5-turbo-1106"
temperature = 1.0
prepare_batch_tasks(df, file_name, categorize_system_prompt, model, temperature)

batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

batch_job_details = client.batches.retrieve(batch_job.id)
print('The batch job id is: ', batch_job_details.id)

print(batch_job_details)

# past batches: 
# movies: batch_mgceyD6JG1QRRMtV84snBap4
# metadata: batch_BZmlZyQBzm0OpnelNaXEhlRs
# metadata with sample ids in task: batch_fzN04mDCeGre54RmssyZmkBH
# batch_BZmlZyQBzm0OpnelNaXEhlRs
# hopefully with sample id s included: batch_UnGVBX6rFxHS3Dv3YzgANIp3
# with all values: batch_YDdgjgKUaZ1fCyq8xoBZje7N
# with all values, gpt 1106: batch_VgezH34KLKyW1XPLqjKTnECH

# Checking batch status
batch_job = client.batches.retrieve(batch_job_details.id)  
print(batch_job)





# Retrieving results 
result_file_id = batch_job.output_file_id
print(result_file_id)







import json
import csv
import os
from openai import OpenAI


def init_openai_client(api_key_path):
    """Initialize the OpenAI client with API key."""
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)


def retrieve_and_save_results(client, output_file_id, output_file_path):
    """Retrieve results from OpenAI and save them to a local file."""
    try:
        result = client.files.content(output_file_id).content
        with open(output_file_path, 'wb') as file:
            file.write(result)
        return True
    except Exception as e:
        print(f"Failed to retrieve or save results: {str(e)}")
        return False

def load_results_from_file(file_path):
    """Load results from a JSONL file."""
    results = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                json_object = json.loads(line.strip())
                results.append(json_object)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    return results

def convert_jsonl_to_csv(input_jsonl_path, output_csv_path):
    """Convert JSONL file to CSV with specified fields."""
    try:
        with open(input_jsonl_path, 'r') as jsonl_file, open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sample_id', 'biome_label', 'geo_location', 'keywords', 'sub-biome'])
            for line in jsonl_file:
                json_obj = json.loads(line)
                content_str = json_obj['response']['body']['choices'][0]['message']['content']
                
                # Ensure the content string is properly converted to a JSON object
                content_data = json.loads(content_str) if isinstance(content_str, str) else content_str
                
                writer.writerow([
                    content_data['sample_id'],
                    content_data['biome_label'],
                    content_data['geo_location'],
                    content_data['keywords'],
                    content_data['sub-biome']
                ])
        print("Data has been processed and written to CSV.")
    except Exception as e:
        print(f"Failed to convert JSONL to CSV: {str(e)}")



# Paths and client initialization
api_key_file = os.path.expanduser("~/my_api_key")
client = init_openai_client(api_key_file)


result_file_id = result_file_id  # This should come from your batch job details
result_file_name = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_job_results_metadata.jsonl"

if retrieve_and_save_results(client, result_file_id, result_file_name):
    results = load_results_from_file(result_file_name)
    print(results)  


current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
filename = f"gpt_clean_output_model{model}_temp{temperature}_{result_file_id}_dt{current_datetime}.txt"
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
output_path = os.path.join(work_dir, filename)
convert_jsonl_to_csv(result_file_name, output_path)











# =============================================================================
# 
# 
# # Retrieving results 
# result_file_id = batch_job.output_file_id
# print(result_file_id)
# 
# result = client.files.content(result_file_id).content
# result_file_name = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_job_results_metadata.jsonl"
# 
# with open(result_file_name, 'wb') as file:
#     file.write(result)
# 
# # Loading data from saved file
# results = []
# with open(result_file_name, 'r') as file:
#     for line in file:
#         # Parsing the JSON string into a dict and appending to the list of results
#         json_object = json.loads(line.strip())
#         results.append(json_object)
#         
# print(results)
# 
# 
# 
# 
# # convert json results to csv: 
# import json
# import csv
# 
# # Define the input JSONL file and output CSV file paths
# input_jsonl_path = result_file_name
# output_csv_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_job_results_metadata.csv"
# 
# # Open the JSONL file and the CSV file for writing
# with open(input_jsonl_path, 'r') as jsonl_file, open(output_csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # Write the CSV header
#     writer.writerow(['col_0', 'col_1', 'col_2', 'col_3', 'col_4'])
#     
#     # Read each line from the JSONL file
#     for line in jsonl_file:
#         json_obj = json.loads(line)  # Parse the line as JSON
#         # Navigate to the nested 'content' field
#         content_str = json_obj['response']['body']['choices'][0]['message']['content']
#         print(content_str)
#         
#         # Parse the JSON string from 'content'
#         content_data = json.loads(content_str)
#         print(content_data)
#         
#         # Extract the desired fields
#         sample_id = content_data['sample_id'] 
#         biome_label = content_data['biome_label']
#         geo_location = content_data['geo_location']  
#         keywords = content_data['keywords']
#         subbiome = content_data['sub-biome']
#         
#         # Write to the CSV file
#         writer.writerow([sample_id, biome_label, geo_location, keywords, subbiome])
# 
# print("Data has been processed and written to CSV.")
# 
# 
# 
# 
# 
# 
# =============================================================================










