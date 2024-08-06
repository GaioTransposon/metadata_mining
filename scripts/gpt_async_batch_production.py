#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:23:58 2024

@author: dgaio
"""


import argparse
import os
import pickle
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from io import BytesIO

from openai import OpenAI


def setup_logging():
    # Get the directory where the script is located
    directory = os.path.dirname(os.path.realpath(__file__))
    # Create a log filename with a timestamp in the script's directory
    log_filename = os.path.join(directory, datetime.now().strftime("gpt_async_batch_production_%Y%m%d%H%M%S.log"))
    
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename, mode='a'),  # Logs to a file in the script's directory
                            logging.StreamHandler()  # Also logs to the standard console output
                        ])


# Initialize OpenAI Client
def init_openai_client(api_key_path):
    with open(api_key_path, "r") as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)

# Load System Prompt
def load_system_prompt(system_prompt_file):
    with open(system_prompt_file, 'r') as file:
        return file.read().strip()

# Fetch Metadata
def fetch_metadata(sample_ids, directory_with_split_metadata, work_dir):
    metadata_dict = {}
    for sample_id in sample_ids:
        folder_name = f"dir_{sample_id[-3:]}"
        folder_path = os.path.join(work_dir, directory_with_split_metadata, folder_name)
        metadata_file_path = os.path.join(folder_path, f"{sample_id}_clean.txt")
        try:
            with open(metadata_file_path, 'r') as file:
                metadata_dict[sample_id] = file.read()
        except Exception as e:
            logging.error(f"Failed to fetch metadata for sample {sample_id}: {e}")
    return metadata_dict

# Save Metadata
def save_metadata(metadata_dict, output_pkl_path):
    with open(output_pkl_path, 'wb') as file:
        pickle.dump(metadata_dict, file)
    logging.info(f"Metadata saved to {output_pkl_path}")

# Prepare Batch Tasks
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


def submit_batch_tasks(client, tasks, batch_info_file, work_dir_full, n_samples, n_batches, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    # Convert tasks to JSONL format (one JSON object per line)
    tasks_jsonl = "\n".join(json.dumps(task) for task in tasks)

    # Convert string to bytes and save to BytesIO for submission
    tasks_buffer = BytesIO(tasks_jsonl.encode('utf-8'))

    # Create and submit batch job
    batch_file = client.files.create(file=tasks_buffer, purpose="batch")
    batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
    logging.info(f"Submitted batch job ID: {batch_job.id}")

    # Write the JSONL data to a file in the specified work directory
    jsonl_filename = f"batch_tasks_metadata_{batch_job.id}.jsonl"
    jsonl_file_path = os.path.join(work_dir_full, jsonl_filename)
    with open(jsonl_file_path, 'w') as file:
        file.write(tasks_jsonl)

    # Log the batch job information
    batch_info = {
        "batch_job_id": batch_job.id,
        "n_samples": n_samples,
        "n_batches": n_batches,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "datetime": datetime.now().strftime('%Y%m%d%H%M')
    }
    update_batch_info_file(os.path.join(work_dir_full, batch_info_file), batch_info)



def update_batch_info_file(info_filename, batch_info):
    if not os.path.exists(info_filename):
        with open(info_filename, "w") as file:
            json.dump([batch_info], file, indent=2)  # Create new file with batch info if not existing
    else:
        with open(info_filename, "r+") as file:
            data = json.load(file)
            data.append(batch_info)  # Append new batch info
            file.seek(0)
            json.dump(data, file, indent=2)



# Load Metadata
def load_metadata(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Get Current Batch Range
def get_current_batch_range(state_file, total_samples, samples_per_batch):
    if os.path.exists(state_file):
        with open(state_file, 'r') as file:
            state_data = json.load(file)
            last_sample = state_data['last_sample']
    else:
        last_sample = 0
    
    start = last_sample
    end = min(last_sample + samples_per_batch, total_samples)
    return start, end

# Update State File
def update_state_file(state_file, end):
    with open(state_file, 'w') as file:
        json.dump({"last_sample": end}, file)

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Integrated script to handle metadata and OpenAI batch processing.")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory path")
    parser.add_argument("--sample_list_file", type=str, required=True, help="File containing list of samples")
    parser.add_argument("--directory_with_split_metadata", type=str, required=True, help="Directory with split metadata")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output .pkl file path")
    parser.add_argument("--system_prompt_file", type=str, required=True, help="System prompt file path")
    parser.add_argument("--api_key_path", type=str, required=True, help="API key file path")
    parser.add_argument("--model", type=str, required=True, help="Model to use for batch processing")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature for completion randomness")
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum number of tokens per completion")
    parser.add_argument("--top_p", type=float, required=True, help="Top p for nucleus sampling")
    parser.add_argument("--frequency_penalty", type=float, required=True, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, required=True, help="Presence penalty")
    parser.add_argument("--n_samples", type=int, default=3500, help="Number of samples per batch")
    parser.add_argument("--n_batches", type=int, default=1, help="Number of batches to send")
    parser.add_argument("--delay_minutes", type=float, default=1.5, help="Delay in minutes between sending batches")
    parser.add_argument("--state_file", type=str, default="batch_state.json", help="File to save the state of batches processed")
    return parser.parse_args()




def main():
    args = parse_args()
    setup_logging()
    
    work_dir_full = os.path.join(os.path.expanduser('~'), args.work_dir)
    
    sample_list_path = os.path.join(work_dir_full, args.sample_list_file)
    with open(sample_list_path, 'r') as file:
        all_samples = [line.strip() for line in file]

    total_samples = len(all_samples)
    samples_per_batch = args.n_samples
    delay_minutes = args.delay_minutes
    state_file = os.path.join(work_dir_full, args.state_file)
    batch_info_file = "batch_job_info_production.json"  # Name of the batch info file
    
    start_index = get_current_batch_range(state_file, total_samples, samples_per_batch)[0]
    batches_processed = 0

    while start_index < total_samples and batches_processed < args.n_batches:
        end_index = min(start_index + samples_per_batch, total_samples)
        selected_samples = all_samples[start_index:end_index]
        
        # Process selected samples
        metadata_dict = fetch_metadata(selected_samples, args.directory_with_split_metadata, work_dir_full)
        save_metadata(metadata_dict, os.path.join(work_dir_full, args.output_pkl))
        
        # Initialize OpenAI client and load system prompt
        client = init_openai_client(os.path.expanduser(args.api_key_path))
        system_prompt = load_system_prompt(os.path.join(os.path.expanduser('~'), args.system_prompt_file))

        # Load metadata and prepare tasks
        metadata_dict = load_metadata(os.path.join(work_dir_full, args.output_pkl))
        df = pd.DataFrame(list(metadata_dict.items()), columns=['sample_id', 'metadata'])
        tasks = prepare_batch_tasks(df, system_prompt, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)

        # Submit batch tasks and wait for delay
        submit_batch_tasks(client, tasks, batch_info_file, work_dir_full, samples_per_batch, args.n_batches, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
        time.sleep(delay_minutes * 60)  # Delay between batches

        start_index += samples_per_batch
        batches_processed += 1
        update_state_file(state_file, end_index)
        logging.info(f"Batch {batches_processed} processed. Total {end_index} samples processed so far.")

if __name__ == "__main__":
    main()



        



python /Users/dgaio/github/metadata_mining/scripts/gpt_async_batch_production.py \
    --work_dir "MicrobeAtlasProject" \
    --sample_list_file "samples_list.txt" \
    --directory_with_split_metadata "sample.info_split_dirs" \
    --output_pkl "metadataprov.pkl" \
    --system_prompt_file "/Users/dgaio/github/metadata_mining/source_data/openai_system_prompt_batch.txt" \
    --api_key_path "my_api_key_production_run" \
    --model "gpt-3.5-turbo-0125" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 \
    --n_samples 3500 \
    --n_batches 77 \
    --delay_minutes 1.5 \
    --state_file "state_file.txt"
    
    
    

    
    
