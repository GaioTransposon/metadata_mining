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
    parser.add_argument('--max_tokens', type=int, required=True, help='we should set the maximum: 4096')
    parser.add_argument('--top_p', type=float, required=True, help='Top-p setting for the GPT model')
    parser.add_argument('--frequency_penalty', type=float, required=True, help='Frequency penalty setting for the GPT model')
    parser.add_argument('--presence_penalty', type=float, required=True, help='Presence penalty setting for the GPT model')
    
    return parser.parse_args()

def main():
    
    setup_logging()
    args = parse_arguments()  # Assumes the parse_arguments function is defined at the top level

    # Phase 1: Metadata Processing
    metadata_processor = MetadataProcessor(args.work_dir, args.input_gold_dict, args.n_samples_per_biome, args.chunk_size, args.system_prompt_file, args.encoding_name, args.seed, args.directory_with_split_metadata)
    
    
    chunks = metadata_processor.run()

    # Phase 2: GPT Interaction
    gpt_interactor = GPTInteractor(args.work_dir, args.n_samples_per_biome, args.chunk_size, args.system_prompt_file, args.encoding_name, args.api_key_path, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty)
    gpt_interactor.run(chunks)

    # Phase 3: Parsing GPT Output
    gpt_parser = GPTOutputParsing(gpt_interactor)
    parsed_df = gpt_parser.run()
    print(parsed_df)

if __name__ == "__main__":
    main()
    
    
    
    
# 20231206 (14:20)
# gpt-3.5-turbo-1106
# 4 samples per biome 

# 20231206 (14:30)
# to get idea of how it counts tokens
# 1 samples per biome 

# 20231207 (13:35)
# to get idea of how it counts tokens
# 1 samples per biome 

# python /Users/dgaio/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 1 \
#     --chunk_size 1000 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --system_prompt_file "openai_system_prompt.txt" \
#     --encoding_name "cl100k_base" \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5

