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

# ----------------------------
# Configuration
# ----------------------------
api_key_path = "/Users/danielagaio/Desktop/keys/my_api_key_production_run"
csv_file_path = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/gpt_matching_ready_dataset.csv'
batch_info_dir = os.path.dirname(csv_file_path)

# GPT model and parameters (same as your script)
model = "gpt-3.5-turbo-0125"
temperature = 1.0
max_tokens = 4096
top_p = 0.75
frequency_penalty = 0.25
presence_penalty = 1.5

samples_per_task = 15  # Number of samples per API call

# ----------------------------
# Initialize OpenAI client
# ----------------------------
with open(api_key_path, "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

# ----------------------------
# Load and sample the dataset
# ----------------------------
df = pd.read_csv(csv_file_path)
assert {'sample_id', 'sub_biome', 'merged'}.issubset(df.columns), "CSV is missing required columns."

# Limit to first 1000 for testing
df = df.head(1000)

# ----------------------------
# Prepare tasks (batching samples into groups inside each task)
# ----------------------------
def prepare_batch_tasks(df, samples_per_task):
    tasks = []
    grouped = [df.iloc[i:i + samples_per_task] for i in range(0, df.shape[0], samples_per_task)]
    for batch_num, group in enumerate(grouped):
        sample_dict = {}
        for _, row in group.iterrows():
            sample_dict[row['sample_id']] = {
                "host_taxonomic_info": row['merged'],
                "sub_biome_description": row['sub_biome']
            }
        
        user_content = f"""
You will be presented with multiple samples.

Each sample includes:
- The sample ID.
- The host taxonomic information (e.g., taxonomic name or related host identifiers).
- The sub-biome description (e.g., body part or environment).

Your task is:
For each sample, determine if the host is clearly mentioned or implied in the sub-biome description.
Reply strictly as a JSON object where the keys are the sample_ids and the values are "Yes" or "No".

Now here is the dictionary of samples:
{json.dumps(sample_dict, indent=2)}

Please reply only with the JSON object. Do not explain or add anything else.
""".strip()

        task = {
            "custom_id": f"check-host-batch-{batch_num}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that replies strictly as JSON."},
                    {"role": "user", "content": user_content}
                ],
            }
        }
        tasks.append(task)
    return tasks

# ----------------------------
# Submit tasks using your fast logic
# ----------------------------
def submit_batch_tasks(client, tasks, batch_info_dir):
    # Convert to JSONL
    tasks_jsonl = "\n".join(json.dumps(task) for task in tasks)
    tasks_buffer = BytesIO(tasks_jsonl.encode('utf-8'))

    batch_file = client.files.create(file=tasks_buffer, purpose="batch")
    batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
    logging.info(f"Submitted batch job ID: {batch_job.id}")

    # Save JSONL file
    jsonl_filename = f"gpt_host_check_tasks_{batch_job.id}.jsonl"
    jsonl_file_path = os.path.join(batch_info_dir, jsonl_filename)
    with open(jsonl_file_path, 'w') as file:
        file.write(tasks_jsonl)

    # Save batch info file
    batch_info = {
        "batch_job_id": batch_job.id,
        "n_samples": len(tasks) * samples_per_task,
        "tasks_created": len(tasks),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "datetime": datetime.now().strftime('%Y%m%d%H%M')
    }
    batch_info_file = os.path.join(batch_info_dir, "batch_job_info_gpt_host_check.json")
    if not os.path.exists(batch_info_file):
        with open(batch_info_file, "w") as file:
            json.dump([batch_info], file, indent=2)
    else:
        with open(batch_info_file, "r+") as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    data.append(batch_info)
                else:
                    data = [data, batch_info]
            except json.JSONDecodeError:
                data = [batch_info]
            file.seek(0)
            json.dump(data, file, indent=2)
            file.truncate()

    logging.info(f"Batch submission completed and info saved in {batch_info_dir}.")

# ----------------------------
# Execute workflow
# ----------------------------
tasks = prepare_batch_tasks(df, samples_per_task)
submit_batch_tasks(client, tasks, batch_info_dir)










import os
import json
import pandas as pd
from datetime import datetime
import logging
from io import BytesIO
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# Configuration
# ----------------------------
api_key_path = "/Users/danielagaio/Desktop/keys/my_api_key_production_run"
csv_file_path = '/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/gpt_matching_ready_dataset.csv'
batch_info_dir = os.path.dirname(csv_file_path)

# GPT model and parameters
model = "gpt-3.5-turbo-0125"
temperature = 0.0
max_tokens = 2048
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0

samples_per_task = 20  # Number of samples per API call

# ----------------------------
# Initialize OpenAI client
# ----------------------------
with open(api_key_path, "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

# ----------------------------
# Load dataset (must contain only unmatched partials you want to check)
# ----------------------------
df = pd.read_csv(csv_file_path)
df = df.head(200)
assert {'sample_id', 'sub_biome', 'merged'}.issubset(df.columns), "CSV is missing required columns."



# ----------------------------
# Prepare batch tasks and keep local mapping
# ----------------------------
all_sample_id_to_question = {}
tasks = []

grouped = [df.iloc[i:i + samples_per_task] for i in range(0, df.shape[0], samples_per_task)]
for batch_num, group in enumerate(grouped):
    questions = []
    for _, row in group.iterrows():
        question = f"Does the string '{row['merged']}' refer to the environment '{row['sub_biome']}'?"
        questions.append(question)
        all_sample_id_to_question[row['sample_id']] = question

    user_content = f"""
Please answer the following questions strictly as a JSON object where the key is the question text exactly as provided, and the value is "Yes" or "No".

Questions:
{json.dumps(questions, indent=2)}

Reply ONLY as JSON object. Do not explain or add anything else.
""".strip()

    task = {
        "custom_id": f"subbiome-check-batch-{batch_num}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that replies strictly as JSON."},
                {"role": "user", "content": user_content}
            ],
        }
    }

    tasks.append(task)

# ----------------------------
# Submit batch tasks
# ----------------------------
tasks_jsonl = "\n".join(json.dumps(task) for task in tasks)
tasks_buffer = BytesIO(tasks_jsonl.encode('utf-8'))

batch_file = client.files.create(file=tasks_buffer, purpose="batch")
batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
logging.info(f"Submitted batch job ID: {batch_job.id}")

# Save JSONL file
jsonl_filename = f"gpt_subbiome_questionstyle_tasks_{batch_job.id}.jsonl"
jsonl_file_path = os.path.join(batch_info_dir, jsonl_filename)
with open(jsonl_file_path, 'w') as file:
    file.write(tasks_jsonl)

# Save local mapping (sample_id → question) for later response parsing
mapping_file = os.path.join(batch_info_dir, f"sample_id_to_question_{batch_job.id}.json")
with open(mapping_file, 'w') as file:
    json.dump(all_sample_id_to_question, file, indent=2)

logging.info(f"Saved sample_id to question mapping to {mapping_file}")
logging.info(f"Batch submission completed and info saved in {batch_info_dir}.")









