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
import glob  

def init_openai_client(api_key_path):
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)

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

def convert_jsonl_content_to_csv(jsonl_content, output_csv_path):
    lines = jsonl_content.splitlines()
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_id', 'biome_label', 'geo_location', 'keywords', 'sub_biome'])
        for line in lines:
            json_obj = json.loads(line)
            content_data = json.loads(json_obj['response']['body']['choices'][0]['message']['content'])

            # because gpt sometimes forgets...
            sub_biome_key = 'sub_biome' if 'sub_biome' in content_data else 'sub-biome'

            writer.writerow([
                content_data['sample_id'],
                content_data['biome_label'],
                content_data['geo_location'],
                content_data['keywords'],
                content_data[sub_biome_key]
            ])
  
            
            

def get_existing_batch_ids(directory):
    pattern = f"{directory}/gpt_clean_output*batch*.csv"
    files = glob.glob(pattern)
    existing_ids = set()
    for file in files:
        batch_id = file.split('_batch')[-1].split('.csv')[0]
        existing_ids.add('batch_' + batch_id) 
    return existing_ids


# Main execution
api_key_file = os.path.expanduser("~/my_api_key")
client = init_openai_client(api_key_file)

directory = "MicrobeAtlasProject" # /Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject
existing_batch_ids = get_existing_batch_ids(directory)
print(existing_batch_ids)

# Load batch job information including model, temperature, and batch_job_id
with open(f"{directory}/batch_job_info.json", "r") as f:
    batch_info_list = json.load(f)

for batch_info in batch_info_list:
    batch_job_id = batch_info["batch_job_id"]
    print('batch_job_id', batch_job_id)
    if batch_job_id not in existing_batch_ids:
        result_json = retrieve_results(client, batch_job_id)
        if result_json:
            
            output_csv_path = f"{directory}/gpt_clean_output_nspb{batch_info['nspb']}_chunking{batch_info['chunking']}_chunksize{batch_info['chunksize']}_model{batch_info['model']}_temp{batch_info['temperature']}_maxtokens{batch_info['max_tokens']}_topp{batch_info['top_p']}_freqp{batch_info['frequency_penalty']}_presp{batch_info['presence_penalty']}_rs{batch_info['rs']}_batch{batch_job_id.split('_')[-1]}_dt{batch_info['datetime']}.csv"
            convert_jsonl_content_to_csv(result_json, output_csv_path)
            print("Batch completed and results saved to CSV.")
        else:
            print("Please wait until the batch job is completed.")
    else:
        print(f"CSV file for batch {batch_job_id} already exists. Skipping creation.")















