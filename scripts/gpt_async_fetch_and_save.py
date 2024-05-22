#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:07:58 2024

@author: dgaio
"""

import json
import csv
import os
from openai import OpenAI


# Initialize the OpenAI client
def init_openai_client(api_key_path):
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)


# Retrieve results and save them locally
def retrieve_results(client, batch_job_id):
    batch_job = client.batches.retrieve(batch_job_id)
    print('\n', batch_job, '\n')
    result_file_id = batch_job.output_file_id
    if result_file_id is not None: 
        result = client.files.content(result_file_id).content
        return result.decode('utf-8')  # Decode bytes to string
    else:
        print('Batch not completed yet')
        return None
    

# Convert JSONL content directly to CSV
def convert_jsonl_content_to_csv(jsonl_content, output_csv_path):
    lines = jsonl_content.splitlines()  # Split the content into lines
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_id', 'biome_label', 'geo_location', 'keywords', 'sub_biome'])
        for line in lines:
            json_obj = json.loads(line)
            content_data = json.loads(json_obj['response']['body']['choices'][0]['message']['content'])
            writer.writerow([
                content_data['sample_id'],
                content_data['biome_label'],
                content_data['geo_location'],
                content_data['keywords'],
                content_data['sub-biome']
            ])



# Main execution
api_key_file = os.path.expanduser("~/my_api_key")
client = init_openai_client(api_key_file)

# Load batch job information including model, temperature, and batch_job_id
with open("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_job_info.json", "r") as f:
    batch_info = json.load(f)

batch_job_id = batch_info["batch_job_id"]
model = batch_info["model"]
temperature = batch_info["temperature"]
top_p = batch_info["top_p"]
frequency_penalty = batch_info["frequency_penalty"]
presence_penalty = batch_info["presence_penalty"]

result_json = retrieve_results(client, batch_job_id)

if result_json:
    # Determine output CSV file name using model, temperature, and batch job ID
    output_csv_path = f"/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_model{model}_temp{temperature}_topp{top_p}_freqp{frequency_penalty}_presp{presence_penalty}_{batch_job_id}.csv"
    convert_jsonl_content_to_csv(result_json, output_csv_path)
    print("Batch completed and results saved to csv.")
else:
    print("Please wait until the batch job is completed.")













