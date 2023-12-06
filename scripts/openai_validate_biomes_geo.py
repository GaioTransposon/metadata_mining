#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:00:38 2023

@author: dgaio
"""

import os
import openai
import pandas as pd
import numpy as np
from collections import Counter
from transformers import GPT2Tokenizer
import argparse
import pickle
import re
from datetime import datetime
import time
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the pipeline.')

    parser.add_argument('--work_dir', type=str, required=True, help='Working directory path')
    parser.add_argument('--input_gold_dict', type=str, required=True, help='Input gold dictionary filename')
    parser.add_argument('--n_samples_per_biome', type=int, required=True, help='how many samples per biome you want to pick?')
    parser.add_argument('--chunk_size', type=int, required=True, help='Number of tokens per chunk.')
    parser.add_argument('--seed', type=int, required=True, help='choose a seed for the random shuffling of the samples e.g.: 42')
    parser.add_argument('--directory_with_split_metadata', type=str, required=True, help='Directory with split metadata')
    parser.add_argument('--api_key_path', type=str, required=True, help='Path to the OpenAI API key')
    parser.add_argument('--model', type=str, required=True, help='GPT model to use')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature setting for the GPT model')
    parser.add_argument('--top_p', type=float, required=True, help='Top-p setting for the GPT model')
    parser.add_argument('--frequency_penalty', type=float, required=True, help='Frequency penalty setting for the GPT model')
    parser.add_argument('--presence_penalty', type=float, required=True, help='Presence penalty setting for the GPT model')
    
    return parser.parse_args()



# =======================================================
# PHASE 0: set up a logging system 
# =======================================================

# Define a logging function that logs to both the console and a file

class CustomFormatter(logging.Formatter):
    MAX_LENGTH = 250  # Set this to your desired length

    def format(self, record):
        if record.levelno == logging.DEBUG and len(record.msg) > self.MAX_LENGTH:
            record.msg = record.msg[:self.MAX_LENGTH] + "..."
        return super().format(record)

def setup_logging():
    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Define the log filename with a timestamp
    log_filename = datetime.now().strftime("openai_validate_biomes_%Y%m%d_%H%M%S.log")
    
    # Join the directory with the log filename to get the full path
    log_filepath = os.path.join(script_directory, log_filename)

    # Set up the basic logging configuration for the console
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.WARNING)  # WARNING, ERROR, and CRITICAL are printed
    
    formatter = CustomFormatter('%(asctime)s [%(levelname)s]: %(message)s')

    # File handler for logging with the full path
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    # Console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    
    


# =======================================================
# PHASE 1: Metadata Processing
# =======================================================
   
class MetadataProcessor:
    
    def __init__(self, work_dir, input_gold_dict, n_samples_per_biome, chunk_size, seed, directory_with_split_metadata):
        self.work_dir = work_dir
        self.input_gold_dict = os.path.join(work_dir, input_gold_dict)
        self.n_samples_per_biome = n_samples_per_biome
        self.seed = seed
        self.chunk_size = chunk_size
        self.directory_with_split_metadata = os.path.join(work_dir, directory_with_split_metadata)
        

    def load_gold_dict(self):
        with open(self.input_gold_dict, 'rb') as file:
            input_gold_dict = pickle.load(file)
            return input_gold_dict[0]  # because the second item is the list of pmids

    def transform_gold_dict_to_df(self, input_gold_dict):
        gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
        gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
        gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])

        # Check if geo_coordinates and geo_text exist in the tuple and extract them
        gold_dict_df['geo_coordinates'] = gold_dict_df['tuple_data'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
        gold_dict_df['geo_text'] = gold_dict_df['tuple_data'].apply(lambda x: x[3] if len(x) > 3 else np.nan)

        gold_dict_df.drop(columns='tuple_data', inplace=True)
        return gold_dict_df

    def get_random_samples(self, gold_dict_df): 
        # Filter out 'unknown' biomes before sampling - at the moment we don't want to test/validate gpt for the classification of "unknown"
        filtered_df = gold_dict_df[gold_dict_df['curated_biome'] != 'unknown']
        random_samples = filtered_df.groupby('curated_biome').apply(lambda x: x.sample(n=self.n_samples_per_biome, random_state=self.seed)).reset_index(drop=True)
        return random_samples

    def fetch_metadata_from_sample(self, sample):
        folder_name = f"dir_{sample[-3:]}"
        folder_path = os.path.join(self.directory_with_split_metadata, folder_name)
        metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
        with open(metadata_file_path, 'r') as f:
            return f.read()

    def process_metadata(self, samples):
        shuffled_samples = samples.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        processed_samples_count = 0
        processed_samples_list = []  # To keep track of which samples have been processed
        
        metadata_dict = {}
        
        for _, row in shuffled_samples.iterrows():
            metadata = self.fetch_metadata_from_sample(row['sample'])
            
            print("Metadata for", row['sample'])
            
            processed_samples_count += 1
            processed_samples_list.append(row['sample'])
            print(f"Processed samples count: {processed_samples_count}")        
            
            cleaned_metadata_lines = []
            for line in metadata.splitlines():
                stripped_line = line.strip()  # strip whitespace
                cleaned_metadata_lines.append(stripped_line)
            cleaned_metadata = "\n".join(cleaned_metadata_lines)
            metadata_dict[row['sample']] = cleaned_metadata
            
            print("Cleaned metadata:")
            print(cleaned_metadata)
            print("===================================")
            
        logging.info(f"All processed samples: {processed_samples_list}")
        
        
        return metadata_dict

    def token_count(self, text):
        """Return the number of tokens in the text (count tokens the BPE way)."""
        # len(text.split()
        
        # load tokenizer 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Tokenize the text and count the tokens
        tokens = tokenizer.tokenize(text)
        
        return len(tokens)

    def create_and_save_chunks(self, metadata_dict):
        max_tokens = self.chunk_size
        chunks = []
        current_chunk, current_token_count = [], 0
        current_chunk_samples = 0  # Track the number of samples in the current chunk
        max_samples_per_chunk = 0  # Track the maximum number of samples across all chunks
    
        logging.info(f"Max tokens allowed per chunk: {max_tokens}")

        too_large_samples = []  # Step 1: Initialize an empty list to store samples that are too large

        for sample_id, metadata in metadata_dict.items():
            item = f"'sample_ID={sample_id}': '{metadata}'"
            item_tokens = self.token_count(item)
    
            print(f"Processing sample: {sample_id} with {item_tokens} tokens")  # Debugging output
    
            # Check if this item alone exceeds the max_tokens
            if item_tokens > max_tokens:
                logging.info(f"Item {sample_id} is too large to fit in a single chunk.")
                too_large_samples.append(sample_id)
                continue
    
            # If adding this item doesn't exceed the token limit, add it to current chunk
            if (current_token_count + item_tokens) <= max_tokens:
                current_chunk.append(item)
                current_token_count += item_tokens
                current_chunk_samples += 1  # Increment the sample count for the current chunk
            else:
                print(f"Chunk is full with {current_token_count} tokens. Saving and starting a new one.")  # Debugging output
                chunks.append('\n~~~\n'.join(current_chunk))  # Use ~~~ as a separator between samples in the same chunk
                max_samples_per_chunk = max(max_samples_per_chunk, current_chunk_samples)  # Update the maximum
                current_chunk, current_token_count = [item], item_tokens
                current_chunk_samples = 1  # Reset for the new chunk (this item is the first one)
    
        # Handle the last chunk
        if current_chunk:
            chunks.append('\n~~~\n'.join(current_chunk))
            max_samples_per_chunk = max(max_samples_per_chunk, current_chunk_samples)  # Update the maximum if needed
    
        logging.info(f"Number of chunks: {len(chunks)}")
        logging.info(f"The maximum number of items in a chunk is: {max_samples_per_chunk}")
        logging.info(f"Samples that exceeded chunk size: {too_large_samples}")
        
        # Get the current date and time
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d%H%M')
    
        # Create the filename
        filename = os.path.join(self.work_dir, f"metadata_chunks_{formatted_time}.txt")
    
        # Write the chunks to the file
        with open(filename, 'w') as f:
            for chunk in chunks:
                f.write(chunk)
                f.write("\n\n-----\n\n")  # Separator between chunks
    
        return chunks


    def run(self):
        gold_dict = self.load_gold_dict()
        gold_dict_df = self.transform_gold_dict_to_df(gold_dict)
        random_samples = self.get_random_samples(gold_dict_df)
        metadata_dict = self.process_metadata(random_samples)
        chunks = self.create_and_save_chunks(metadata_dict)
        return chunks
    
# =======================================================
# PHASE 2: GPT Interaction
# =======================================================

class GPTInteractor:
   
    def __init__(self, work_dir, n_samples_per_biome, chunk_size, api_key_path, model, temperature, top_p, frequency_penalty, presence_penalty):
        self.work_dir = work_dir
        self.n_samples_per_biome = n_samples_per_biome
        self.chunk_size = chunk_size
        self.api_key_path = api_key_path
        self.api_key = self.load_api_key()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.saved_filename = None  # This will store the filename once saved


    def token_count(self, text):
        """Return the number of tokens in the text."""
        return len(text.split())
    
    
    def consolidate_chunks_to_strings(self, chunks):
        content_strings = []
        chunk_tokens = []  # This will store the number of tokens in each chunk
        
        for i, chunk in enumerate(chunks, 1):
            total_tokens = self.token_count(chunk)
            print(f"Chunk {i} Content (Total Tokens: {total_tokens})")
            logging.info(f"Chunk {i} Content (Total Tokens: {total_tokens})")
            content_strings.append(chunk)  
            chunk_tokens.append(total_tokens)  # Store the number of tokens
            #print(f"Chunk {i} Content:")
            #print("----")
        
        print(content_strings)
        return content_strings, chunk_tokens


    def load_api_key(self):
        try:
            with open(self.api_key_path, "r") as file:
                api_key = file.read().strip()
                return api_key.strip()
        except FileNotFoundError:
            print(f"File '{self.api_key_path}' not found.")
            return None
        except IOError:
            print(f"Error reading file '{self.api_key_path}'.")
            return None

    def gpt_request(self, content_string):
        openai.api_key = self.api_key

        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": '''
                    Based on the metadata texts below, you have to:
                        1) guess where the sample each metadata text is based on, comes from. Your choices are: 'animal' (includes human), 'plant', 'water', 'soil'. Give strictly 1-word answer for each sample ID.
                        2) extract the location in terms of coordinates, and in terms of text. When info is not available, write 'NA'. 
                        All values separated by '__'. An example of the answer for a sample: SRS123456__animal__12.37 N 1.51 W__Burkina Faso
                    '''
                },
                {
                    "role": "user",
                    "content": content_string
                }
            ],
            temperature=self.temperature,
            # max_tokens left to default which is 4096 
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

    def interact_with_gpt(self, content_strings, chunk_tokens):
        """
        Iterate over content_strings and make requests to GPT.

        Parameters:
        - content_strings: List of string contents to process.

        Returns:
        - List of responses from GPT-3.
        """
        
        gpt_responses = []
        TOKEN_LIMIT = 9500  # Adding a buffer
        tokens_processed_in_last_minute = 0
        last_request_time = time.time()

        for content_string, tokens in zip(content_strings, chunk_tokens):
            # If adding the next chunk exceeds the limit, wait
            if tokens_processed_in_last_minute + tokens > TOKEN_LIMIT:
                elapsed_time = time.time() - last_request_time
                wait_time = 60 - elapsed_time

                if wait_time > 0:
                    logging.info(f"Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

                tokens_processed_in_last_minute = 0  # Reset token count
                last_request_time = time.time()  # Update timestamp

            # Send request to API
            try:
                response = self.gpt_request(content_string=content_string)
                gpt_responses.append(response)
                tokens_processed_in_last_minute += tokens
            except openai.error.OpenAIError as e:
                if "rate limit" in str(e).lower():
                    logging.info("Rate limit exceeded. Waiting for 2 minutes...")
                    time.sleep(120)
                else:
                    logging.error(f"Error encountered: {e}")

        return gpt_responses


    def save_gpt_responses_to_file(self, gpt_responses):
        """
        Save the content of GPT responses to a file.
    
        Parameters:
        - gpt_responses: List of GPT responses.
    
        Returns:
        - None
        """
        # Extract the "content" from each response with error handling
        contents = []
        for response in gpt_responses:
            try:
                contents.append(response['choices'][0]['message']['content'])
            except KeyError:
                contents.append("ERROR: Malformed response")
        
        # Join all contents with a separator (two newlines for readability)
        final_content = "\n\n".join(contents)
        
        # Construct the filename
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
        self.saved_filename = f"gpt_raw_output_nspb{self.n_samples_per_biome}_chunksize{self.chunk_size}_model{self.model}_temp{self.temperature}_topp{self.top_p}_freqp{self.frequency_penalty}_presp{self.presence_penalty}_dt{current_datetime}.txt"
        self.saved_filename = os.path.join(self.work_dir, self.saved_filename)
    
        # Write to the file
        with open(self.saved_filename, 'w') as file:
            file.write(final_content)
    
        logging.info(f"Saved GPT responses to: {self.saved_filename}")

    def get_saved_filename(self):
        """ 
        Returns the path of the saved file containing the GPT responses.

        Returns:
        - Path to the saved file.
        """
        if self.saved_filename:
            return self.saved_filename
        else:
            print("No file has been saved yet!")
            return None
    
    def run(self, chunks):
        content_strings, chunk_tokens = self.consolidate_chunks_to_strings(chunks)
        logging.info("Starting interaction with GPT...")
        gpt_responses = self.interact_with_gpt(content_strings, chunk_tokens)  # Pass chunk_tokens here
        logging.info("Finished interaction with GPT.")
        
        self.save_gpt_responses_to_file(gpt_responses) 
    



# =======================================================
# PHASE 3: GPT Output Parsing
# =======================================================
class GPTOutputParsing:

    def __init__(self, interactor_instance):
        self.filepath = interactor_instance.get_saved_filename()
        self.raw_content = self.load_from_file()
        self.parsed_data = None
        self.clean_filename = None  

    def load_from_file(self):
        if self.filepath:
            try:
                with open(self.filepath, 'r') as file:
                    return file.read().splitlines()  
            except FileNotFoundError:
                logging.error(f"File '{self.filepath}' not found.")
                return None
            except IOError:
                logging.error(f"Error reading file '{self.filepath}'.")
                return None
        else:
            logging.error("No filepath provided.")
            return None

    def parse_samples(self):
        result = {}
        pattern = re.compile(r'(SRS|ERS|DRS)\d+__\w+__.*?__.*')
        
        for line in self.raw_content:
            match = pattern.match(line)
            if match:
                parts = line.split('__')
                if len(parts) == 4:
                    sample_id, biome, coordinates, location = parts
                    result[sample_id] = {
                        'biome': biome,
                        'geo_coordinates': coordinates,
                        'geo_text': location
                    }

        return result

    def prepare_dataframe(self, parsed_data_dict):
        return pd.DataFrame.from_dict(parsed_data_dict, orient='index').reset_index().rename(columns={'index': 'sample'})

    def save_cleaned_to_file(self):
        self.clean_filename = self.filepath.replace('gpt_raw_output', 'gpt_clean_output')
        self.parsed_data.to_csv(self.clean_filename, index=False)
        logging.info(f"Saved clean GPT output to: {self.clean_filename}")
        
    def run(self):
        parsed_samples = self.parse_samples()
        self.parsed_data = self.prepare_dataframe(parsed_samples)
        self.save_cleaned_to_file()
        return self.parsed_data


# =======================================================
# Main Execution
# =======================================================

def main():
    
    setup_logging()
    args = parse_arguments()  # Assumes the parse_arguments function is defined at the top level

    # Phase 1: Metadata Processing
    metadata_processor = MetadataProcessor(args.work_dir, args.input_gold_dict, args.n_samples_per_biome, args.chunk_size, args.seed, args.directory_with_split_metadata)
    chunks = metadata_processor.run()

    # Phase 2: GPT Interaction
    gpt_interactor = GPTInteractor(args.work_dir, args.n_samples_per_biome, args.chunk_size, args.api_key_path, args.model, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
    gpt_interactor.run(chunks)

    # Phase 3: Parsing GPT Output
    gpt_parser = GPTOutputParsing(gpt_interactor)
    parsed_df = gpt_parser.run()
    print(parsed_df)

if __name__ == "__main__":
    main()
    
    
    
# 20231205 (17:20)
# gpt-3.5-turbo-1106
# 4 samples per biome 
# chunk_size: 1200
# temperatures: 1.00
# top_p: 0.75
# frequency_penalty: 0.25
# presence penalty: 1.50


# python /Users/dgaio/github/metadata_mining/scripts/openai_validate_biomes_geo.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 20 \
#     --chunk_size 1200 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --model "gpt-3.5-turbo-1106" \
#     --temperature 1.00 \
#     --top_p 0.75 \
#     --frequency_penalty 0.25 \
#     --presence_penalty 1.5
    