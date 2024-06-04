#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:36:00 2024

@author: dgaio
"""

# NB: works with latets openai (not 0.28)
# pip uninstall openai
# pip install openai 

import argparse
import json
import os
import pandas as pd
import pickle
from openai import OpenAI
from datetime import datetime
from io import BytesIO
import re
    
    

def init_openai_client(api_key_path):
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)


def load_system_prompt(system_prompt_file):
    with open(system_prompt_file, 'r') as file:
        return file.read().strip()
    

def prepare_batch_tasks(df, system_prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
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
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
            }
        }
        tasks.append(task)
    return tasks


def parse_filename(filename):
    pattern = r"metadataprov_nspb(\d+)_chunking(\w+)_chunksize(\d+)_rs(\d+)"
    match = re.search(pattern, filename)
    if match:
        return {
            "nspb": match.group(1),
            "chunking": match.group(2),
            "chunksize": match.group(3),
            "rs": match.group(4),
        }
    return {}


def find_latest_file(directory, prefix, file_extension):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(file_extension)]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    print('latest file is:', latest_file)
    return latest_file




def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenAI GPT batch processing with custom parameters.")
    parser.add_argument("--work_dir", required=True, help="Working directory for files")
    parser.add_argument("--system_prompt_file", required=True, help="System prompt file name")
    parser.add_argument("--api_key_path", required=True, help="Path to the OpenAI API key file")
    parser.add_argument("--model", required=True, help="Model to be used for completion")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature for completion randomness")
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum number of tokens per completion")
    parser.add_argument("--top_p", type=float, required=True, help="Top p for nucleus sampling")
    parser.add_argument("--frequency_penalty", type=float, required=True, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, required=True, help="Presence penalty")
    return parser.parse_args()




def main():
    args = parse_args()
    
    client = init_openai_client(os.path.expanduser(args.api_key_path))
    work_dir_full = os.path.join(os.path.expanduser('~'), args.work_dir)
    system_prompt = load_system_prompt(os.path.join(os.path.expanduser('~'), args.system_prompt_file))
    data_file = find_latest_file(work_dir_full, "metadataprov", ".pkl")
    print("Using metadata file:", data_file)
    file_params = parse_filename(data_file)
    
    with open(data_file, 'rb') as file:
        metadata_dict = pickle.load(file)
    df = pd.DataFrame(list(metadata_dict.items()), columns=['sample_id', 'metadata'])
    
    # Assuming metadata has 'sample_id' and 'metadata' information
    tasks = prepare_batch_tasks(df, system_prompt, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
    
    # convert tasks to JSONL format (one JSON object per line)
    tasks_jsonl = "\n".join(json.dumps(task) for task in tasks)
    # convert string to bytes and save to BytesIO
    tasks_buffer = BytesIO(tasks_jsonl.encode('utf-8'))

    # create and submit batch job
    batch_file = client.files.create(file=tasks_buffer, purpose="batch")
    batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
    
    # Metadata file is named after the batch id 
    filename = f"batch_tasks_metadata_{batch_job.id}.jsonl"
    output_file_path = os.path.join(work_dir_full, filename)
    with open(output_file_path, 'w') as file:
        file.write(tasks_jsonl)

    # Store batch info     
    batch_info = {
    "batch_job_id": batch_job.id,
    "nspb": file_params['nspb'],
    "chunking": file_params['chunking'],
    "chunksize": file_params['chunksize'],
    "rs": file_params['rs'],
    "model": args.model,
    "temperature": args.temperature,
    "max_tokens": args.max_tokens,
    "top_p": args.top_p,
    "frequency_penalty": args.frequency_penalty,
    "presence_penalty": args.presence_penalty,
    "datetime": datetime.now().strftime('%Y%m%d%H%M')
    }

    info_filename = os.path.join(work_dir_full, "batch_job_info.json")
    if not os.path.exists(info_filename):
        with open(info_filename, "w") as f:
            json.dump([batch_info], f, indent=2)
    else:
        with open(info_filename, "r+") as f:
            data = json.load(f)
            data.append(batch_info)
            f.seek(0)
            json.dump(data, f, indent=2)
    

if __name__ == "__main__":
    main()
    
    




# python /Users/dgaio/github/metadata_mining/scripts/gpt_async_batch.py \
#     --work_dir "MicrobeAtlasProject" \
#     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 



