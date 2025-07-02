#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:36:00 2024

@author: dgaio
"""


# # run as: 
# python ~/github/metadata_mining/scripts/gpt_async_batch.py \
#     --work_dir "MicrobeAtlasProject" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 5 \
#     --chunking "no" \
#     --chunk_size 3000 \
#     --seed 22 \
#     --directory_with_split_metadata "sample_info_split_dirs" \
#     --system_prompt_file "openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --output_format "inline"


import argparse
import json
import logging
import os
import pickle
import re
import time
from datetime import datetime
from io import BytesIO

import pandas as pd
from openai import OpenAI

# ---- INTERNAL IMPORTS -------------------------------------------------------
# We keep using the standalone helper that already does the heavy lifting
# of selecting N random samples, optional chunking, and writing the resulting
# pickle (metadataprov_*.pkl)
from openai_02_metadata_fetching import MetadataFetching
# -----------------------------------------------------------------------------


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Collect CLI args for BOTH steps in one go."""

    p = argparse.ArgumentParser(
        description="1) fetch / chunk metadata  ➜  2) submit async GPT batch"
    )

    # --- preparation-step flags (formerly metadata_preparation.py) ------------
    p.add_argument("--work_dir", default=".", help="Base working directory")
    p.add_argument("--input_gold_dict", required=True, help="gold_dict.pkl")
    p.add_argument("--n_samples_per_biome", type=int, required=True)
    p.add_argument("--chunking", choices=["yes", "no"], required=True)
    p.add_argument("--chunk_size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--directory_with_split_metadata", required=True)
    p.add_argument("--system_prompt_file", required=True)
    p.add_argument("--encoding_name", required=True)

    # --- batch-submission flags (formerly gpt_async_batch.py) -----------------
    p.add_argument("--api_key_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--max_tokens", type=int, required=True)
    p.add_argument("--top_p", type=float, required=True)
    p.add_argument("--frequency_penalty", type=float, required=True)
    p.add_argument("--presence_penalty", type=float, required=True)
    p.add_argument("--output_format", choices=["json", "inline"], required=True)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions reused from original scripts
# ──────────────────────────────────────────────────────────────────────────────

def init_openai_client(api_key_path: str) -> OpenAI:
    with open(api_key_path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    return OpenAI(api_key=key)


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def prepare_batch_tasks(df: pd.DataFrame, system_prompt: str, *, model: str, temperature: float,
                        max_tokens: int, top_p: float, frequency_penalty: float,
                        presence_penalty: float, output_format: str):
    """Create the JSON bodies for the OpenAI batch API."""
    tasks = []
    for _, row in df.iterrows():
        user_content = f"Sample ID: {row['sample_id']}, Metadata: {row['metadata']}"
        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if output_format == "json":
            body["response_format"] = {"type": "json_object"}

        tasks.append({
            "custom_id": f"task-{row['sample_id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })
    return tasks


def parse_filename(fname: str):
    """Extract parameters from the metadataprov filename."""
    m = re.search(r"metadataprov_nspb(\d+)_chunking(\w+)_chunksize(\d+)_rs(\d+)", fname)
    return dict(zip(["nspb", "chunking", "chunksize", "rs"], m.groups())) if m else {}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # -------- Paths -----------------------------------------------------------
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    gold_dict_path     = os.path.join(work_dir, args.input_gold_dict)
    system_prompt_path = os.path.join(work_dir, args.system_prompt_file)
    api_key_path       = os.path.join(work_dir, args.api_key_path)
    split_dir          = os.path.join(work_dir, args.directory_with_split_metadata)

    # ---------- 1) Metadata fetch / chunk -------------------------------------
    start = time.time()
    fetcher = MetadataFetching(
        work_dir,
        args.directory_with_split_metadata,
        gold_dict_path,
        args.n_samples_per_biome,
        args.chunking,
        args.chunk_size,
        args.seed,
    )
    latest_pickle = fetcher.run()  # we modify MetadataFetching.run() to *return* the filepath
    duration = time.time() - start
    logging.info("Metadata fetching/chunking took %.1f s", duration)

    # ---------- 2) Prepare & submit async batch --------------------------------
    # If the helper didn’t return the file path, fall back to pattern search
    if latest_pickle is None:
        pkls = [f for f in os.listdir(work_dir) if f.startswith("metadataprov_") and f.endswith(".pkl")]
        latest_pickle = max((os.path.join(work_dir, p) for p in pkls), key=os.path.getmtime)

    logging.info("Using metadata file %s", latest_pickle)

    params = parse_filename(os.path.basename(latest_pickle))

    with open(latest_pickle, "rb") as f:
        meta = pickle.load(f)
    df = pd.DataFrame(meta.items(), columns=["sample_id", "metadata"])

    system_prompt = load_system_prompt(system_prompt_path)
    tasks = prepare_batch_tasks(
        df, system_prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        output_format=args.output_format,
    )

    tasks_jsonl = "\n".join(json.dumps(t) for t in tasks)
    buf = BytesIO(tasks_jsonl.encode())

    client = init_openai_client(api_key_path)
    batch_file = client.files.create(file=buf, purpose="batch")
    batch_job  = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")

    # Save task metadata and batch details ------------------------------------
    jsonl_name = f"batch_tasks_metadata_{batch_job.id}.jsonl"
    with open(os.path.join(work_dir, jsonl_name), "w", encoding="utf-8") as f:
        f.write(tasks_jsonl)

    info = {
        **params,
        "batch_job_id": batch_job.id,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "output_format": args.output_format,
        "datetime": datetime.now().strftime("%Y%m%d%H%M"),
    }

    info_path = os.path.join(work_dir, "batch_job_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(info)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logging.info("Batch %s submitted. Metadata & info saved.", batch_job.id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()





# =============================================================================
# # script: gpt_async_batch.py
# 
# import argparse
# import json
# import os
# import pandas as pd
# import pickle
# from openai import OpenAI
# from datetime import datetime
# from io import BytesIO
# import re
#     
#     
# 
# def init_openai_client(api_key_path):
#     with open(api_key_path, "r") as file:
#         api_key = file.read().strip()
#     return OpenAI(api_key=api_key)
# 
# 
# def load_system_prompt(system_prompt_file):
#     with open(system_prompt_file, 'r') as file:
#         return file.read().strip()
#     
# 
# def prepare_batch_tasks(df, system_prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, output_format):
#     tasks = []
#     for _, row in df.iterrows():
#         user_content = f"Sample ID: {row['sample_id']}, Metadata: {row['metadata']}"
#         task = {
#             "custom_id": f"task-{row['sample_id']}",
#             "method": "POST",
#             "url": "/v1/chat/completions",
#             "body": {
#                 "model": model,
#                 "temperature": temperature,
#                 "max_tokens": max_tokens,
#                 "top_p": top_p,
#                 "frequency_penalty": frequency_penalty,
#                 "presence_penalty": presence_penalty,
#                 #"response_format": {"type": "json_object"},
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_content}
#                 ],
#             }
#         }
#         
#         if output_format == 'json':
#             task["body"]["response_format"] = {"type": "json_object"}
#             print('Output will be in json format')
#         elif output_format == 'inline':
#             print('Output will be in inline format')
#         
#         
#         tasks.append(task)
#     return tasks
# 
# 
# def parse_filename(filename):
#     pattern = r"metadataprov_nspb(\d+)_chunking(\w+)_chunksize(\d+)_rs(\d+)"
#     match = re.search(pattern, filename)
#     if match:
#         return {
#             "nspb": match.group(1),
#             "chunking": match.group(2),
#             "chunksize": match.group(3),
#             "rs": match.group(4),
#         }
#     return {}
# 
# 
# def find_latest_file(directory, prefix, file_extension):
#     files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(file_extension)]
#     if not files:
#         return None
#     latest_file = max(files, key=os.path.getmtime)
#     print('latest file is:', latest_file)
#     return latest_file
# 
# 
# 
# 
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run OpenAI GPT batch processing with custom parameters.")
#     parser.add_argument("--work_dir", required=True, help="Working directory for files")
#     parser.add_argument("--system_prompt_file", required=True, help="System prompt file name")
#     parser.add_argument("--api_key_path", required=True, help="Path to the OpenAI API key file")
#     parser.add_argument("--model", required=True, help="Model to be used for completion")
#     parser.add_argument("--temperature", type=float, required=True, help="Temperature for completion randomness")
#     parser.add_argument("--max_tokens", type=int, required=True, help="Maximum number of tokens per completion")
#     parser.add_argument("--top_p", type=float, required=True, help="Top p for nucleus sampling")
#     parser.add_argument("--frequency_penalty", type=float, required=True, help="Frequency penalty")
#     parser.add_argument("--presence_penalty", type=float, required=True, help="Presence penalty")
#     parser.add_argument("--output_format", required=True, help="can be inline or json")
#     return parser.parse_args()
# 
# 
# 
# 
# def main():
#     
#     
#     # -------------------------------------------------
#     # Resolve paths relative to --work_dir
#     # -------------------------------------------------
#     args = parse_args()
#     
#     # In Docker, "." resolves to /MicrobeAtlasProject
#     work_dir = os.path.abspath(args.work_dir)
#     
#     # Treat every filename as **relative** to work_dir
#     api_key_path        = os.path.join(work_dir, args.api_key_path)
#     system_prompt_file  = os.path.join(work_dir, args.system_prompt_file)
#     
#     # Locate the latest metadata-prep pickle produced by metadata_preparation.py
#     # (files look like:  metadataprov_nspb5_chunkingno_chunksize3000_rs22.pkl)
#     data_file = find_latest_file(
#         directory=work_dir,
#         prefix="metadataprov",
#         file_extension=".pkl",
#     )
#     
#     if data_file is None:
#         raise FileNotFoundError("No metadataprov_*.pkl file found in " + work_dir)
#     
#     print("Using metadata file:", data_file)
# 
#     # -------------------------------------------------
#     # OpenAI client + system prompt
#     # -------------------------------------------------
#     client         = init_openai_client(api_key_path)
#     system_prompt  = load_system_prompt(system_prompt_file)
#     
#     # args = parse_args()
#     
#     # client = init_openai_client(os.path.expanduser(args.api_key_path))
#     # work_dir_full = os.path.join(os.path.expanduser('~'), args.work_dir)
#     # system_prompt = load_system_prompt(os.path.join(os.path.expanduser('~'), args.system_prompt_file))
#     # data_file = find_latest_file(work_dir_full, "metadataprov", ".pkl")
#     
#     
#     print("Using metadata file:", data_file)
#     file_params = parse_filename(data_file)
#     
#     with open(data_file, 'rb') as file:
#         metadata_dict = pickle.load(file)
#     df = pd.DataFrame(list(metadata_dict.items()), columns=['sample_id', 'metadata'])
#     
#     # Assuming metadata has 'sample_id' and 'metadata' information
#     tasks = prepare_batch_tasks(df, system_prompt, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty, args.output_format)
#     
#     # convert tasks to JSONL format (one JSON object per line)
#     tasks_jsonl = "\n".join(json.dumps(task) for task in tasks)
#     # convert string to bytes and save to BytesIO
#     tasks_buffer = BytesIO(tasks_jsonl.encode('utf-8'))
# 
#     # create and submit batch job
#     batch_file = client.files.create(file=tasks_buffer, purpose="batch")
#     batch_job = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
#     
#     # Metadata file is named after the batch id 
#     output_file_path = f"batch_tasks_metadata_{batch_job.id}.jsonl"
#     with open(output_file_path, 'w') as file:
#         file.write(tasks_jsonl)
# 
#     # Store batch info     
#     batch_info = {
#     "batch_job_id": batch_job.id,
#     "nspb": file_params['nspb'],
#     "chunking": file_params['chunking'],
#     "chunksize": file_params['chunksize'],
#     "rs": file_params['rs'],
#     "model": args.model,
#     "temperature": args.temperature,
#     "max_tokens": args.max_tokens,
#     "top_p": args.top_p,
#     "frequency_penalty": args.frequency_penalty,
#     "presence_penalty": args.presence_penalty,
#     "output_format": args.output_format,
#     "datetime": datetime.now().strftime('%Y%m%d%H%M')
#     }
# 
#     info_filename = "batch_job_info.json"
#     if not os.path.exists(info_filename):
#         with open(info_filename, "w") as f:
#             json.dump([batch_info], f, indent=2)
#     else:
#         with open(info_filename, "r+") as f:
#             data = json.load(f)
#             data.append(batch_info)
#             f.seek(0)
#             json.dump(data, f, indent=2)
#     
# 
# if __name__ == "__main__":
#     main()
#     
#     
# 
# 
# 
# #####
# # test with json: 
# 
# # python /Users/dgaio/github/metadata_mining/scripts/gpt_async_batch.py \
# #     --work_dir "MicrobeAtlasProject" \
# #     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt_json.txt" \
# #     --api_key_path "Desktop/keys/my_api_key" \
# #     --model "gpt-3.5-turbo-1106" \
# #     --temperature 1.00 \
# #     --max_tokens 4096 \
# #     --top_p 0.75 \
# #     --frequency_penalty 0.25 \
# #     --presence_penalty 1.5 \
# #     --output_format 'json'
# #####
# 
# #####
# # test with inline: 
# 
# # python /Users/dgaio/github/metadata_mining/scripts/gpt_async_batch.py \
# #     --work_dir "MicrobeAtlasProject" \
# #     --system_prompt_file "github/metadata_mining/source_data/openai_system_prompt.txt" \
# #     --api_key_path "Desktop/keys/my_api_key" \
# #     --model "gpt-3.5-turbo-1106" \
# #     --temperature 1.00 \
# #     --max_tokens 4096 \
# #     --top_p 0.75 \
# #     --frequency_penalty 0.25 \
# #     --presence_penalty 1.5 \
# #     --output_format 'inline'
# #####
# =============================================================================











