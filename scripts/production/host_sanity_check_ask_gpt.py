#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:31:19 2025

@author: danielagaio
"""


import os
import json
import pandas as pd
from datetime import datetime
import logging
from io import BytesIO
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
api_key_path = "/Users/danielagaio/Desktop/keys/my_api_key_production_run"
with open(api_key_path, "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

# Load the prepared dataframe from the saved CSV
df = pd.read_csv('/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/gpt_matching_ready_dataset.csv')

# Check expected columns
assert {'sample_id', 'sub_biome', 'merged'}.issubset(df.columns), "CSV is missing required columns."

# Prepare system prompt directly in the script
system_prompt = "You are a helpful assistant. Answer only 'Yes' or 'No'."

# Prepare tasks for batch submission
tasks = []
for _, row in df.iterrows():
    sample_id = row['sample_id']
    merged_text = row['merged']
    sub_biome_text = row['sub_biome']

    user_content = f"""
Given the host taxonomic information:
{merged_text}

Is the host mentioned or clearly implied in this sub-biome description?
{sub_biome_text}

Answer only 'Yes' or 'No'.
    """.strip()

    task = {
        "custom_id": f"check-host-{sample_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "temperature": 0,
            "max_tokens": 3,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        }
    }
    tasks.append(task)

logging.info(f"Prepared {len(tasks)} tasks for batch submission.")

# Submit batch
tasks_jsonl = "\n".join(json.dumps(task) for task in tasks)
tasks_buffer = BytesIO(tasks_jsonl.encode('utf-8'))

batch_file = client.files.create(file=tasks_buffer, purpose="batch")
batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")

logging.info(f"Submitted batch job ID: {batch_job.id}")

# Save JSONL and batch job info for tracking
batch_info_dir = os.path.dirname(os.path.realpath(__file__))
batch_filename = f"gpt_host_check_tasks_{batch_job.id}.jsonl"
with open(os.path.join(batch_info_dir, batch_filename), 'w') as file:
    file.write(tasks_jsonl)

batch_info = {
    "batch_job_id": batch_job.id,
    "n_samples": len(tasks),
    "model": "gpt-4",
    "datetime": datetime.now().strftime('%Y%m%d%H%M')
}
with open(os.path.join(batch_info_dir, "batch_job_info_gpt_host_check.json"), "w") as file:
    json.dump(batch_info, file, indent=2)

logging.info(f"Batch submission completed and info saved.")
