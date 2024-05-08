#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:25:08 2024

@author: dgaio
"""


import aiohttp
import asyncio
import glob
import os
import json
import argparse
import openai

class GPTInteractor:
    def __init__(self, api_key, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    async def fetch_gpt_response(self, session, prompt):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': self.model,
            'prompt': prompt,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }
        async with session.post('https://api.openai.com/v1/completions', headers=headers, json=payload) as response:
            return await response.json()

    async def process_chunks(self, file_path):
        async with aiohttp.ClientSession() as session:
            with open(file_path, 'r') as file:
                chunks = file.read().split("\n\n-----\n\n")
            results = []
            tasks = [self.fetch_gpt_response(session, chunk) for chunk in chunks if chunk.strip()]
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
            return results

def find_latest_chunk_file(directory):
    list_of_files = glob.glob(os.path.join(directory, 'metadata_chunks_*.txt'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process metadata chunks with GPT-3 asynchronously.')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--model', type=str, required=True, help='GPT model to use')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature setting for the GPT model')
    parser.add_argument('--max_tokens', type=int, required=True, help='Maximum number of tokens')
    parser.add_argument('--top_p', type=float, required=True, help='Top-p setting for the GPT model')
    parser.add_argument('--frequency_penalty', type=float, required=True, help='Frequency penalty setting')
    parser.add_argument('--presence_penalty', type=float, required=True, help='Presence penalty setting')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    interactor = GPTInteractor(args.api_key, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
    latest_chunk_file = find_latest_chunk_file(args.work_dir)

    if latest_chunk_file:
        results = await interactor.process_chunks(latest_chunk_file)
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    asyncio.run(main())



python gpt_interact.py \
    --work_dir "MicrobeAtlasProject" \
    --api_key "my_api_key" \
    --model "gpt-3.5-turbo-1106" \
    --temperature 1.00 \
    --max_tokens 4096 \
    --top_p 0.75 \
    --frequency_penalty 0.25 \
    --presence_penalty 1.5 
    
    






