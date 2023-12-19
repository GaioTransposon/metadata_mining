#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:58:00 2023

@author: dgaio
"""


import argparse
from openai_01_setup_and_args import setup_logging
from openai_02_metadata_processing import MetadataProcessor
from openai_03_gpt_interaction import GPTInteractor
from openai_04_gpt_parsing import GPTOutputParsing
import time  


# =======================================================
# Main Execution
# =======================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the pipeline.')

    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--input_gold_dict', type=str, required=True, help='Input gold dictionary filename')
    parser.add_argument('--n_samples_per_biome', type=int, required=True, help='how many samples per biome you want to pick?')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of tokens per chunk.')
    parser.add_argument('--seed', type=int, required=True, help='choose a seed for the random shuffling of the samples e.g.: 42')
    parser.add_argument('--directory_with_split_metadata', type=str, required=True, help='Directory with split metadata')
    parser.add_argument('--system_prompt_file', type=str, required=True, help='it should be named openai_system_prompt.txt (location: work_dir)')
    parser.add_argument('--encoding_name', type=str, required=True, help='name of encoder (for tokenizer) e.g.: cl100k_base')
    parser.add_argument('--api_key_path', type=str, required=True, help='Path to the OpenAI API key')
    parser.add_argument('--model', type=str, required=True, help='GPT model to use')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature setting for the GPT model')
    parser.add_argument('--max_tokens', type=int, required=True, help='we should set the maximum: 4096. some models don t support the maximum. run with few samples to check if max_tokens chosen is appropriate')
    parser.add_argument('--top_p', type=float, required=True, help='Top-p setting for the GPT model')
    parser.add_argument('--frequency_penalty', type=float, required=True, help='Frequency penalty setting for the GPT model')
    parser.add_argument('--presence_penalty', type=float, required=True, help='Presence penalty setting for the GPT model')
    
    return parser.parse_args()


    
def main():
    
    setup_logging()
    args = parse_arguments()  

    # Phase 1: Metadata Processing
    start_time = time.time()
    metadata_processor = MetadataProcessor(args.work_dir, args.input_gold_dict, args.n_samples_per_biome, args.chunk_size, args.system_prompt_file, args.encoding_name, args.seed, args.directory_with_split_metadata)
    metadata_processor.run()
    end_time = time.time() 
    print(f"Metadata Processing time: {end_time - start_time} seconds")
    
    # Phase 2: GPT Interaction
    start_time = time.time()
    gpt_interactor = GPTInteractor(args.work_dir, args.n_samples_per_biome, args.chunk_size, args.system_prompt_file, args.api_key_path, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
    gpt_interactor.run()
    end_time = time.time() 
    print(f"GPT Interaction time: {end_time - start_time} seconds")
    
    # Phase 3: Parsing GPT Output
    start_time = time.time()
    gpt_parser = GPTOutputParsing(gpt_interactor, args.encoding_name)
    parsed_df = gpt_parser.run()
    end_time = time.time() 
    print(f"Parsing GPT Output time: {end_time - start_time} seconds")
    
    print(parsed_df)


if __name__ == "__main__":
    main()
    
    
    
    
# 20231206 (14:20)
# gpt-3.5-turbo-1106
# 4 samples per biome 

# 20231206 (14:30)
# to get idea of how it counts tokens
# 1 samples per biome 

# 20231207 (18:22)
# chunk size 1200
# 40 samples per biome 

# 20231214 (16:16)
# lots of tests --> spent $0.05

# 20231214 (17:55)
# chunk size 1500
# "gpt-3.5-turbo-1106"
# 40 samples per biome 

# 20231214 (17:57)
# chunk size 1500
# "gpt-3.5-turbo-0613"
# 40 samples per biome 
# max_tokens 2000
# not finished bad error gate 

# 20231214 (18:08)
# chunk size 1500
# "gpt-3.5-turbo-1106"
# 40 samples per biome 
# to test costs 
# what my log file says: 
# Total input tokens (including system prompt(s)): 65587
# Total output tokens: 3614
# what openai states: 
# cost: $0.09
# requests: 45
# input tokens: 66587
# output tokens: 3571

# 20231219 (14:00)
# 200 samples per biome
# to test tokens and cost
#Input: 323788 
#Completion: 18166 

# 20231219 (16:50)
# 2 samples per biome
# to test new prompt
#Input: 323788 
#Completion: 18166 

# python /Users/dgaio/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 2 \
#     --chunk_size 1500 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "openai_system_prompt2.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5





