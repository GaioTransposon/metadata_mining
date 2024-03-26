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
import pandas as pd


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
    parser.add_argument('--opt_text', type=str, required=False, help='extra text to indicate special run deets')    
    
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
    gpt_interactor = GPTInteractor(args.work_dir, args.n_samples_per_biome, args.chunk_size, args.seed, args.system_prompt_file, args.api_key_path, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty, args.opt_text)
    gpt_interactor.run()
    end_time = time.time() 
    print(f"GPT Interaction time: {end_time - start_time} seconds")
    
    # Phase 3: Parsing GPT Output
    start_time = time.time()
    gpt_parser = GPTOutputParsing(gpt_interactor, args.encoding_name, metadata_processor.processed_sample_ids)
    parsed_df, missing_sample_ids = gpt_parser.run()
    end_time = time.time() 
    print(f"Parsing GPT Output time: {end_time - start_time} seconds")
    
    print('parsed_df before adding missing samples:', parsed_df)
    
    print('missing_sample_ids: ', missing_sample_ids)
    
    
    # Maximum number of retries
    max_retries = 3
    # Initialize the retry counter
    retry_count = 0

    while missing_sample_ids and retry_count < max_retries:
        # Increment the retry counter
        retry_count += 1
        print(f"Retry attempt {retry_count}")
    
        missing_samples_df = pd.DataFrame(list(missing_sample_ids), columns=['sample'])
        
        ###
        specific_metadata_dict = metadata_processor.process_metadata(missing_samples_df)
        specific_chunks = metadata_processor.create_and_save_chunks(specific_metadata_dict, metadata_processor.encoding_name)
        gpt_responses = gpt_interactor.interact_with_gpt(specific_chunks)
        parsed_df_for_specific_samples = gpt_parser.parse_direct_responses(gpt_responses)
        combined_parsed_df = pd.concat([parsed_df, parsed_df_for_specific_samples]).reset_index(drop=True)
        gpt_parser.parsed_data = combined_parsed_df  # Update the parsed_data attribute of the gpt_parser instance
        gpt_parser.save_cleaned_to_file()  # Save the combined DataFrame using the existing method
    
        print('combined_parsed_df after : ', combined_parsed_df)
        
        combined_parsed_df_ids = set(combined_parsed_df['col_0'].unique())
        print('combined_parsed_df_ids: ', len(combined_parsed_df_ids))
        missing_sample_ids =  missing_sample_ids - combined_parsed_df_ids
        print('missing_sample_ids: ', missing_sample_ids)


    ###
    
    
    print('missing_sample_ids after all retries : ', missing_sample_ids)
    
    # After all processing and reprocessing
    print(f"Total API requests made: {gpt_interactor.get_api_request_count()}")



if __name__ == "__main__":
    main()
    
    
    
    

# eventually...
# =============================================================================
# try:
#     metadata_processor.run()
# except Exception as e:
#     logging.error(f"Error during metadata processing: {e}")
# 
# try:
#     gpt_interactor.run()
# except Exception as e:
#     logging.error(f"Error during GPT interaction: {e}")
# =============================================================================
    
    

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

# 20231221 (14:00)
# 200 samples per biome (incl unknown)

# 20231221 (17:50)
# 4 samples per biome 
# test editing prompt to ask table format 

# 20240103
# 200 samples per biome 
# testing prompt with "lake" vs "river" as suggested sub-biome. 
# 16:10 "lake"
# 16:23 "river"
# no answer because problem parsing output. solved. 

# 20240103
# 200 samples per biome 
# testing prompt with "lake" vs "river" as suggested sub-biome. 
# 17:45 "river" --> 982/984 rows 984 because 16 too large chunks
# 18:00 "lake" --> 983/984 rows 984 because 16 too large chunks

# 20240104
# few samples per biome -  various tests
# testing new prompt to extract only coordinates
# when using "openai_system_prompt_coordinates.txt", use chunk_size 400

# 20240104 15:32 
# 200 nspb, "openai_system_prompt_coordinates.txt", chunk_size 400

# 20240117
# 200 nspb, "openai_system_prompt_coordinates.txt" 


# 20240313
# 200 nspb, "openai_system_prompt.txt"
# seed 42 vs 22 vs 11

# 20240314
# 200 nspb
# testing prompts: with 1,2, and 4 examples for format (openai_system_prompt.txt vs openai_system_prompt_2examples.txt vs openai_system_prompt_4examples.txt)

# 20240318
# 100 nspb; chunk_size 2000
# random seed: 42 vs 24

# 20240319
# 200 nspb
# chunk_size 1500 vs 2000 vs 3000 vs 5000 vs 6000 vs 8000
# 1500: 17 samples too large to fit in chunk --> test_chunking_bias.py not working -  too few samples...? 
# 2000: 1 samples too large to fit in chunk 

# 20240326
# opt_text: "normal" vs "please" 

# python /Users/dgaio/github/metadata_mining/scripts/openai_main.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 100 \
#     --chunk_size 2000 \
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
#     --presence_penalty 1.5 \
#     --opt_text "normal"





