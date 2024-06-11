#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:07:58 2024

@author: dgaio
"""

import json
import csv
import os
import glob
from openai import OpenAI

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

def convert_jsonl_content_to_csv(jsonl_content, output_csv_path, failed_samples_path):
    lines = jsonl_content.splitlines()
    with open(output_csv_path, 'w', newline='') as csvfile, open(failed_samples_path, 'a') as failed_file:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_id', 'biome_label', 'geo_location', 'keywords', 'sub_biome'])
        for line in lines:
            try:
                json_obj = json.loads(line)
                content_data = json.loads(json_obj['response']['body']['choices'][0]['message']['content'])

                writer.writerow([
                    content_data.get('sample-id', 'N/A'),
                    content_data.get('biome-label', 'N/A'),
                    content_data.get('geo-location', 'N/A'),
                    content_data.get('keywords', 'N/A'),
                    content_data.get('sub-biome', 'N/A')
                ])
            except Exception as e:
                # Log the line and the exception to a failure log
                failed_file.write(f"Failed to process line: {line}\nError: {str(e)}\n")

def get_existing_batch_ids(directory):
    pattern = f"{directory}/gpt_clean_output*batch*.csv"
    files = glob.glob(pattern)
    existing_ids = set()
    for file in files:
        batch_id = file.split('_batch')[-1].split('.csv')[0]
        existing_ids.add('batch_' + batch_id) 
    return existing_ids

def log_failed_batch(directory, batch_job_id):
    failed_log_path = os.path.join(directory, "failed_async_batches.txt")
    with open(failed_log_path, "a") as file:
        file.write(batch_job_id + "\n")

# Main execution
api_key_file = os.path.expanduser("~/my_api_key")
client = init_openai_client(api_key_file)

directory = "MicrobeAtlasProject"  # Example directory
failed_samples_path = os.path.join(directory, "failed_async_samples.txt")

existing_batch_ids = get_existing_batch_ids(directory)
print(existing_batch_ids)

# Load batch job information
with open(f"{directory}/batch_job_info.json", "r") as f:
    batch_info_list = json.load(f)

for batch_info in batch_info_list:
    batch_job_id = batch_info["batch_job_id"]
    print('batch_job_id', batch_job_id)
    if batch_job_id not in existing_batch_ids:
        try:
            result_json = retrieve_results(client, batch_job_id)
            if result_json:
                output_csv_path = f"{directory}/gpt_clean_output_nspb{batch_info['nspb']}_chunking{batch_info['chunking']}_chunksize{batch_info['chunksize']}_model{batch_info['model']}_temp{batch_info['temperature']}_maxtokens{batch_info['max_tokens']}_topp{batch_info['top_p']}_freqp{batch_info['frequency_penalty']}_presp{batch_info['presence_penalty']}_rs{batch_info['rs']}_batch{batch_job_id.split('_')[-1]}_dt{batch_info['datetime']}.csv"
                convert_jsonl_content_to_csv(result_json, output_csv_path, failed_samples_path)
                print("Batch completed and results saved to CSV.")
            else:
                print("Please wait until the batch job is completed.")
        except Exception as e:
            print(f"Error processing batch {batch_job_id}: {str(e)}. Logging failed batch.")
            log_failed_batch(directory, batch_job_id)
    else:
        print(f"CSV file for batch {batch_job_id} already exists. Skipping creation.")






