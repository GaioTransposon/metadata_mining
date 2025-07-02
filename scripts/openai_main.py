#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:58:00 2023

@author: dgaio
"""



# NB: in order to run it needs `pip install openai`

import argparse
import time
import pandas as pd
from datetime import datetime
import os
import logging
from openai_01_setup_and_args import setup_logging
from openai_02_metadata_fetching import MetadataFetching
from openai_02_metadata_processing import MetadataProcessor
from openai_03_gpt_interaction import GPTInteractor
#from openai_04_gpt_parsing import GPTOutputParsing




# =======================================================
# Main Execution
# =======================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the pipeline.')

    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--input_gold_dict', type=str, required=True, help='Input gold dictionary filename')
    parser.add_argument('--n_samples_per_biome', type=int, required=True, help='how many samples per biome you want to pick?')
    parser.add_argument('--chunking', type=str, choices=['yes', 'no'], required=True, help='Enable or disable chunking of metadata for GPT requests.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of tokens per chunk.')
    parser.add_argument('--seed', type=int, required=True, help='choose a seed for the random shuffling of the samples e.g.: 42')
    parser.add_argument('--directory_with_split_metadata', type=str, required=True, help='Directory with split metadata')
    parser.add_argument('--system_prompt_file', type=str, required=True, help='it should be named openai_system_prompt.txt. Remember to change the input prompt based on the output_format')
    parser.add_argument('--encoding_name', type=str, required=True, help='name of encoder (for tokenizer) e.g.: cl100k_base')
    parser.add_argument('--api_key_path', type=str, required=True, help='Path to the OpenAI API key')
    parser.add_argument('--model', type=str, required=True, help='GPT model to use')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature setting for the GPT model')
    parser.add_argument('--max_tokens', type=int, required=True, help='we should set the maximum: 4096. some models don t support the maximum. run with few samples to check if max_tokens chosen is appropriate')
    parser.add_argument('--top_p', type=float, required=True, help='Top-p setting for the GPT model')
    parser.add_argument('--frequency_penalty', type=float, required=True, help='Frequency penalty setting for the GPT model')
    parser.add_argument('--presence_penalty', type=float, required=True, help='Presence penalty setting for the GPT model')
    parser.add_argument('--max_requests_per_minute', type=float, required=True, help='set the max RPM')
    parser.add_argument('--opt_text', type=str, required=False, help='extra text to indicate special run deets')   
    parser.add_argument('--output_format', type=str, choices=['inline', 'json'], required=True, help='output format for GPT responses. Remember to change the input prompt based on this')
    
    return parser.parse_args()


def main():
    


    # -------------------------------------------------
    # Resolve all important paths (relative to work_dir)
    # -------------------------------------------------
    args = parse_arguments()
    
    # The script runs from /MicrobeAtlasProject in Docker, so "." = CWD
    work_dir = os.path.abspath(args.work_dir)
    
    # Assume all other paths are relative to work_dir
    input_gold_dict     = os.path.join(work_dir, args.input_gold_dict)
    system_prompt_file  = os.path.join(work_dir, args.system_prompt_file)
    api_key_path        = os.path.join(work_dir, args.api_key_path)
    
    # (Optional) Debug output
    print("work_dir         :", work_dir)
    print("input_gold_dict  :", input_gold_dict)
    print("system_prompt_file:", system_prompt_file)
    print("api_key_path     :", api_key_path)

    
    # Phase 0: set up a logging system 
    setup_logging()
    
    
    # Phase 1: Metadata Fetching
    start_time = time.time()
    metadata_fetcher = MetadataFetching(work_dir, args.directory_with_split_metadata, input_gold_dict, args.n_samples_per_biome, args.chunking, args.chunk_size, args.seed)
    metadata_fetcher.run()
    end_time = time.time() 
    print(f"Metadata fetching time: {end_time - start_time} seconds")
    logging.info(f"Metadata fetching time: {end_time - start_time} seconds")
    
    
    # Phase 2: Metadata Processing 
    start_time = time.time()
    metadata_processor = MetadataProcessor(work_dir, args.chunking, args.chunk_size, system_prompt_file, args.encoding_name)
    processed_metadata = metadata_processor.process_metadata()
    chunks, complete_sample_ids = metadata_processor.create_and_save_chunks(processed_metadata, return_ids=True)
    metadata_processor.save_chunks_to_file(chunks) 
    end_time = time.time() 
    print(f"Metadata Processing time: {end_time - start_time} seconds")
    logging.info(f"Metadata Processing time: {end_time - start_time} seconds")
    

    print('complete_sample_ids', len(complete_sample_ids))
    complete_sample_ids =set(complete_sample_ids)
    
    
    # PHASE 3: GPT Interaction
    start_time = time.time()
    gpt_interactor = GPTInteractor(work_dir, system_prompt_file, api_key_path, args.model, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty, args.presence_penalty, args.max_requests_per_minute)
    responses = gpt_interactor.get_gpt_responses()
    print('####### responses ###############################################################')
    print(responses)
 
    gpt_interactor.save_gpt_responses_to_file(responses)
    end_time = time.time() 
    print(f"GPT Interaction time: {end_time - start_time} seconds")
    logging.info(f"GPT Interaction time: {end_time - start_time} seconds")


    # Phase 4: Parsing GPT Output
    if args.output_format == 'json':
        from openai_04_gpt_parsing_json import GPTOutputParsing as GPTOutputParsingJSON
        parser = GPTOutputParsingJSON(work_dir)
    else:
        from openai_04_gpt_parsing_inline import GPTOutputParsing as GPTOutputParsingInline
        parser = GPTOutputParsingInline(work_dir)
        
    start_time = time.time()
    main_parsed_df = parser.run(responses)
    print('========================================================== ')
    print(main_parsed_df)
    
    parsed_sample_ids = set(main_parsed_df['col_0'].unique())

    missing_samples = list(complete_sample_ids - parsed_sample_ids)
    end_time = time.time() 
    print(f"Parsing GPT Output time: {end_time - start_time} seconds")
    logging.info(f"Parsing GPT Output time: {end_time - start_time} seconds")
    
    print('parsed_df before adding missing samples:', main_parsed_df)


    retry_count = 0
    max_retries = 3
    
    print('##########')
    print(f"missing samples after {retry_count}th retry: {missing_samples}")   
    logging.info(f"missing samples after {retry_count}th retry: {missing_samples}")
    

    #missing_samples = ['SRS5495722', 'SRS1416741', 'SRS2920130', 'ERS4232978', 'SRS1761000', 'ERS2363505', 'SRS6079931', 'ERS3333693']
    while missing_samples and retry_count < max_retries:
        retry_count += 1
        print('##########')
        print(f"Retry attempt {retry_count}")
        
        # Convert missing_samples to set for set operation
        missing_samples_set = set(missing_samples)
        
        processed_metadata = metadata_processor.process_metadata(missing_samples)
        chunks = metadata_processor.create_and_save_chunks(processed_metadata)
        metadata_processor.save_chunks_to_file(chunks) 

        responses = gpt_interactor.get_gpt_responses()
        #print(responses)
        gpt_interactor.save_gpt_responses_to_file(responses)
            
        new_parsed_df = parser.run(responses)
        
        new_parsed_sample_ids = set(new_parsed_df['col_0'].unique()) if 'col_0' in new_parsed_df.columns else set()
        missing_samples_set -= new_parsed_sample_ids
        missing_samples = list(missing_samples_set)  

        # append parsed_df to main_parsed_df
        if 'col_0' in new_parsed_df.columns:
            main_parsed_df = pd.concat([main_parsed_df, new_parsed_df], axis=0)
        
              
    my_tot_api_count = gpt_interactor.get_api_request_count()
    print('my_tot_api_count', my_tot_api_count)
    logging.info(f"my_tot_api_count: {my_tot_api_count}")

    # save final df to file:
    current_datetime = datetime.now().strftime('%Y%m%d%H%M')
    filename = f"gpt_clean_output_nspb{args.n_samples_per_biome}_chunking{args.chunking}_chunksize{args.chunk_size}_model{args.model}_temp{args.temperature}_maxtokens{args.max_tokens}_topp{args.top_p}_freqp{args.frequency_penalty}_presp{args.presence_penalty}_rs{args.seed}_API{my_tot_api_count}_{args.opt_text}_dt{current_datetime}.txt"
    output_path = os.path.join(work_dir, filename)
    main_parsed_df.to_csv(output_path, index=False)
    logging.info(f"Saved clean GPT output to: {output_path}")
    
    print(f"missing samples after all retries: {missing_samples}")   
    logging.info(f"missing samples after all retries: {missing_samples}")
    print('parsed_df after adding all missing samples:', main_parsed_df)
    
    
    
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

# 20240412
# chunking vs no chunking

# 20240415
# tests on Orion: 
# MicrobeAtlasProject
# "cloudstor/Gaio/MicrobeAtlasProject" 

# 20240502
# testing if it still works (after gold dict is now no longer a tuple) 
# nspb 2, 20, 


# 20241008 
# running using openai_system_prompt_batch.txt 

# 20241014
# running testing new param --format 

# 20250304
# running with better prompt

# 20250630
# test for container 

# python github/metadata_mining/scripts/openai_main.py \
#     --work_dir . \
#     --input_gold_dict gold_dict.pkl \
#     --n_samples_per_biome 5 \
#     --chunking no \
#     --chunk_size 2000 \
#     --seed 22 \
#     --directory_with_split_metadata sample_info_split_dirs \
#     --system_prompt_file openai_system_better_prompt_json.txt \
#     --encoding_name cl100k_base \
#     --api_key_path my_api_key \
#     --model gpt-3.5-turbo-1106 \
#     --temperature 1.00 \
#     --max_tokens 4096 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5 \
#     --max_requests_per_minute 3500 \
#     --opt_text normal \
#     --output_format json








