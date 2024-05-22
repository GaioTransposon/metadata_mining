#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:36:00 2024

@author: dgaio
"""


import json
import os
import pandas as pd
from openai import OpenAI

# Initialize the OpenAI client
def init_openai_client(api_key_path):
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)


# Load the system prompt
def load_system_prompt(work_dir, system_prompt_file):
    prompt_file = os.path.join(work_dir, system_prompt_file)
    with open(prompt_file, 'r') as file:
        return file.read().strip()


# Prepare batch tasks
def prepare_batch_tasks(df, output_file_path, system_prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
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
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
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

        
# Main execution
api_key_file = os.path.expanduser("~/my_api_key")
client = init_openai_client(api_key_file)
work_dir = "/Users/dgaio/github/metadata_mining/source_data"
prompt_file_name = "openai_system_prompt_batch.txt"
categorize_system_prompt = load_system_prompt(work_dir, prompt_file_name)
data_file = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_prov.csv"
df = pd.read_csv(data_file)

model = "gpt-3.5-turbo-1106"
temperature = 1.0
max_tokens = 4096
top_p = 0.75
frequency_penalty = 0.25
presence_penalty = 1.5
file_name = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_tasks_metadata.jsonl"
prepare_batch_tasks(df, file_name, categorize_system_prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)

# Create and submit batch job
batch_file = client.files.create(file=open(file_name, "rb"), purpose="batch")
batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")


# Save batch job ID with model and temperature for retrieval
batch_info = {
    "batch_job_id": batch_job.id,
    "model": model,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": top_p,
    "frequency_penalty": frequency_penalty,
    "presence_penalty": presence_penalty
}

with open("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/batch_job_info.json", "w") as f:
    json.dump(batch_info, f, indent=2)






