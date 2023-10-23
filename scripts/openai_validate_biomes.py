#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:41:38 2023

@author: dgaio
"""

import os
import openai
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import pickle
import re
from datetime import datetime
import time



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
        metadata_file_path = os.path.join(folder_path, f"{sample}.txt")
        with open(metadata_file_path, 'r') as f:
            return f.read()

    def process_metadata(self, samples):
        shuffled_samples = samples.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        processed_samples_count = 0
        processed_samples_list = []  # To keep track of which samples have been processed
        
        # endings to filter out ("$" makes sure it will skip these if preceeded by a character or empty space)
        endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
        metadata_dict = {}
        
        for _, row in shuffled_samples.iterrows():
            metadata = self.fetch_metadata_from_sample(row['sample'])  # change this line as it's now a method
            
            print("Metadata for", row['sample'])
            
            processed_samples_count += 1
            processed_samples_list.append(row['sample'])
            print(f"Processed samples count: {processed_samples_count}")        
            
            cleaned_metadata_lines = []
            for line in metadata.splitlines():
                stripped_line = line.strip()  # strip whitespace
                should_keep = True
                if stripped_line.lower().startswith(("experiment", "run", ">")):
                    should_keep = False
                else:
                    for ending in endings_to_remove:
                        if re.search(ending, stripped_line, re.IGNORECASE):
                            print(f"Rejected line (ends with {ending}): {stripped_line}")
                            should_keep = False
                            break
                if should_keep:
                    cleaned_metadata_lines.append(stripped_line)
            cleaned_metadata = "\n".join(cleaned_metadata_lines)
            metadata_dict[row['sample']] = cleaned_metadata
            
            print("Cleaned metadata:")
            print(cleaned_metadata)
            print("===================================")
            
        print(f"All processed samples: {processed_samples_list}")
        
        return metadata_dict


    def token_count(self, text):
        """Return the number of tokens in the text."""
        return len(text.split())

    def create_and_save_chunks(self, metadata_dict):
        max_tokens = self.chunk_size
        chunks = []
        current_chunk, current_token_count = [], 0
        current_chunk_samples = 0  # Track the number of samples in the current chunk
        max_samples_per_chunk = 0  # Track the maximum number of samples across all chunks
    
        print(f"Max tokens allowed per chunk: {max_tokens}")
    
        for sample_id, metadata in metadata_dict.items():
            item = f"'sample_ID={sample_id}': '{metadata}'"
            item_tokens = self.token_count(item)
    
            print(f"Processing sample: {sample_id} with {item_tokens} tokens")  # Debugging output
    
            # Check if this item alone exceeds the max_tokens
            if item_tokens > max_tokens:
                print(f"Item {sample_id} is too large to fit in a single chunk.")
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
    
        print('Number of chunks: ', len(chunks))
        print(f"The maximum number of items in a chunk is: {max_samples_per_chunk}")
    
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

# =============================================================================
#     def create_and_save_chunks(self, metadata_dict):
#         max_tokens = self.chunk_size
#         chunks, current_chunk, current_token_count = [], [], 0
#         
#         print(f"Max tokens allowed per chunk: {max_tokens}")
#     
#         for sample_id, metadata in metadata_dict.items():
#             item = f"'sample_ID={sample_id}': '{metadata}'"
#             item_tokens = self.token_count(item)
#     
#             print(f"Processing sample: {sample_id} with {item_tokens} tokens")  # Debugging output
#     
#             # Check if this item alone exceeds the max_tokens
#             if item_tokens > max_tokens:
#                 print(f"Item {sample_id} is too large to fit in a single chunk.")
#                 continue
#     
#             # If adding this item doesn't exceed the token limit, add it to current chunk
#             if (current_token_count + item_tokens) <= max_tokens:
#                 current_chunk.append(item)
#                 current_token_count += item_tokens
#             else:
#                 print(f"Chunk is full with {current_token_count} tokens. Saving and starting a new one.")  # Debugging output
#                 chunks.append('\n~~~\n'.join(current_chunk))  # Use ~~~ as a separator between samples in the same chunk
#                 current_chunk, current_token_count = [item], item_tokens
#     
#         # Handle the last chunk
#         if current_chunk:
#             chunks.append('\nNEXT SAMPLE -->\n'.join(current_chunk))
#     
#         print('Number of chunks: ', len(chunks))
#         print(f"The maximum number of items in a chunk is: {len(max(chunks, key=lambda x: self.token_count(x)))}")
#     
#         # Get the current date and time
#         current_time = datetime.now()
#         formatted_time = current_time.strftime('%Y%m%d%H%M')
#     
#         # Create the filename
#         filename = os.path.join(self.work_dir, f"metadata_chunks_{formatted_time}.txt")
#     
#         # Write the chunks to the file
#         with open(filename, 'w') as f:
#             for chunk in chunks:
#                 f.write(chunk)
#                 f.write("\n\n-----\n\n")  # Separator between chunks
#     
#         return chunks
# =============================================================================


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
            print(f"Chunk {i} Content (Total Tokens: {total_tokens}):")
            content_strings.append(chunk)  
            chunk_tokens.append(total_tokens)  # Store the number of tokens
            #print(f"Chunk {i} Content:")
            #print("----")
        
        print(content_strings)
        return content_strings, chunk_tokens


# =============================================================================
#     def consolidate_chunks_to_strings(self, chunks):
#         """
#         Consolidate individual items within each chunk into one content string.
# 
#         Parameters:
#         - chunks: List of lists containing the metadata chunked by tokens.
# 
#         Returns:
#         - List of consolidated content strings from each chunk.
#         """
#         # Empty list to store content_strings for each chunk
#         content_strings = []
# 
#         # Lists to store chunk details (can be returned if needed)
#         chunk_tokens = []
# 
#         # Joining the content within each chunk
#         for i, chunk in enumerate(chunks, 1):
#             # Compute the number of tokens in the chunk
#             total_tokens = sum(len(item.split()) for item in chunk)
#             chunk_tokens.append(total_tokens)
#             
#             print(f"Chunk {i} Content (Number of items: {len(chunk)} | Total Tokens: {total_tokens}):")
#             content_string = "\n".join(chunk)
#             content_strings.append(content_string)  # Store the content_string
#             print(f"Chunk {i} Content:")
#             print(content_string)
#             print("----")
#         
#         return content_strings, chunk_tokens  # Return the consolidated content_strings
# =============================================================================
    

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
                    "content": "Based on the metadata texts below, you have to guess where the sample each metadata text is based on, come from. Your choices are are: 'animal' (includes human), 'plant', 'water', 'soil'. Report the sample ID each time and the answer (strictly 1-word answer for each sample ID)."
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
                    print(f"Waiting for {wait_time:.2f} seconds...")
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
                    print("Rate limit exceeded. Waiting for 2 minutes...")
                    time.sleep(120)
                else:
                    print(f"Error encountered: {e}")

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
    
        print(f"Saved GPT responses to: {self.saved_filename}")


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
        print("Starting interaction with GPT...")
        gpt_responses = self.interact_with_gpt(content_strings, chunk_tokens)  # Pass chunk_tokens here
        print("Finished interaction with GPT.")
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
                    return file.readlines()  
            except FileNotFoundError:
                print(f"File '{self.filepath}' not found.")
                return None
            except IOError:
                print(f"Error reading file '{self.filepath}'.")
                return None
        else:
            print("No filepath provided.")
            return None

    def check_keyword_mentions(self):
        keywords = ['animal', 'soil', 'water', 'plant', 'human']
        total_count = sum(line.lower().count(keyword) for line in self.raw_content for keyword in keywords)
        print(f"Total count of mentions: {total_count}")
        return total_count

    def parse_samples(self):
        result = {}
        sample_id_pattern = re.compile(r'(SRS|ERS|DRS)\d+')
        buffer = None
        text = ''   # Initialize the text variable here
        
        for line in self.raw_content:
            id_match = sample_id_pattern.search(line)
            if id_match:
                if buffer:
                    result[buffer] = text.strip()
                buffer = id_match.group()
                text = line[id_match.end():]
            elif buffer:
                text += ' ' + line

        if buffer:
            result[buffer] = text.strip()

        return result

    def check_missing_samples(self, parsed_data):
        def extract_all_sample_ids():
            sample_id_pattern = re.compile(r'(SRS|ERS|DRS)\d+')
            return set(m.group() for line in self.raw_content for m in [sample_id_pattern.search(line)] if m)
        
        all_sample_ids = extract_all_sample_ids()
        missing_samples = all_sample_ids - set(parsed_data.keys())
        print("Number of samples we lost in the parsing: ", missing_samples)
        return missing_samples

    def prepare_dataframe(self, parsed_data_dict):
        parsed_data = pd.DataFrame(list(parsed_data_dict.items()), columns=['sample', 'gpt_generated_output_raw'])

        def extract_clean_output(raw_output):
            raw_output_lower = raw_output.lower()
            for keyword in ['plant', 'animal', 'soil', 'water']:
                if keyword in raw_output_lower:
                    return keyword
            if 'human' in raw_output_lower:
                return 'animal'
            return None

        parsed_data['gpt_generated_output_clean'] = parsed_data['gpt_generated_output_raw'].apply(extract_clean_output)
        self.parsed_data = parsed_data
        return parsed_data
    
    def save_cleaned_to_file(self):
        # Modify the filename to switch from 'raw' to 'clean'
        self.clean_filename = self.filepath.replace('gpt_raw_output', 'gpt_clean_output')
    
        # Save the dataframe to CSV format
        self.parsed_data.to_csv(self.clean_filename, index=False)
        print(f"Saved clean GPT output to: {self.clean_filename}")
        
    def run(self):
        self.check_keyword_mentions()
        parsed_samples = self.parse_samples()
        self.check_missing_samples(parsed_samples)
        df = self.prepare_dataframe(parsed_samples)
        self.save_cleaned_to_file()
        return df
    

# =======================================================
# Main Execution
# =======================================================

def main():
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


# # calling the script: 
# python /Users/dgaio/github/metadata_mining/scripts/openai_validate_biomes.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunk_size 500 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --model "gpt-3.5-turbo-16k-0613" \
#     --temperature 0.25 \
#     --top_p 0.75 \
#     --frequency_penalty 0 \
#     --presence_penalty 0 


# 20231012
# Temperatures: 0.0 0.25 0.5 0.75 1.0
# 10 samples per biome

# 20231013
# Temperatures: 0.0 0.25 0.5 0.75 1.0
# 40 samples per biome
# Cost: 
# prompt: 283170  $0.84951
# completion: 7750  $0.031

# 20231013
# Temperatures: 0.0 0.25 0.5 0.75 1.0
# 100 samples per biome --> 71 requests per temperature = 71*5 = 355
# 1 request = 1 chunk = 1000 tokens max (excl. sys prompt)
# 0.0 ; df rows: 398
# 0.25 ; df rows: 397
# 0.50 ; df rows: 398
# 0.75 ; df rows: 398
# 1.0 ; df rows: 395

# 20231016
# gpt-4 vs gpt-3.5-turbo-16k-0613
# 10 samples per biome (7 requests each model)

# 20231017 (before 13:30)
# gpt-4 vs gpt-3.5-turbo-16k-0613 (now corrected for case sensitivity)
# 10 samples per biome (7 requests each model)

# 20231017 (15:28 and 15:17 and 15:15)
# gpt-3.5-turbo-16k-0613: 1000 vs 500 vs 250 tokens chunks 
# 10 samples per biome

# 20231017 (16:07, 16:09, 16:13)
# gpt-3.5-turbo-16k-0613: 1000 vs 500 vs 250 tokens chunks 
# 20 samples per biome (81 requests * 3 = 243)
# tot input prompt tokens: 109644
# tot output prompt tokens: 4453 


# 20231018 (17:36)
# gpt-3.5-turbo-16k-0613: 
# 40 samples per biome 
# chunk_size: 300 vs 500 vs 700 vs 1000 vs 1400 vs 2100
# Number of chunks:  54 vs 41 vs 34 vs 23 vs 16 vs 10 = tot 178
# maximum # items in a chunk: 6 vs 8 vs 10 vs 12 vs 17 vs 25
# output rows: 145 vs 154 vs 158 vs 160 vs 156 vs 152
# tot input prompt tokens: 316199/1000*0.003= $0.948597
# tot output prompt tokens: 8533/1000*0.004= $0.03413

# 20231019 (16:00)
# gpt-3.5-turbo-16k-0613: 
# 200 samples per biome 
# chunk_size: 300 vs 500 vs 700 vs 1000 vs 1400 vs 2100 vs 3000
# tot input prompt tokens: 2037963/1000*0.003= $6.113889
# tot output prompt tokens: 52231/1000*0.004= $0.208924

# 20231023 (13:00)
# gpt-3.5-turbo-16k-0613: 
# 200 samples per biome 
# chunk_size: 1200
# temperatures: 0.0 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
# tot input prompt tokens: .../1000*0.003= ...
# tot output prompt tokens: .../1000*0.004= ...

# done/doing: 1.00 1.25

# python /Users/dgaio/github/metadata_mining/scripts/openai_validate_biomes.py \
#     --work_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/" \
#     --input_gold_dict "gold_dict.pkl" \
#     --n_samples_per_biome 200 \
#     --chunk_size 1200 \
#     --seed 42 \
#     --directory_with_split_metadata "sample.info_split_dirs" \
#     --api_key_path "/Users/dgaio/my_api_key" \
#     --model "gpt-3.5-turbo-16k-0613" \
#     --temperature 1.25 \
#     --top_p 0.75 \
#     --frequency_penalty 0 \
#     --presence_penalty 0 





# =============================================================================
# def fetch_metadata_from_sample(sample, directory_with_split_metadata):
#     folder_name = f"dir_{sample[-3:]}"
#     folder_path = os.path.join(directory_with_split_metadata, folder_name)  
#     metadata_file_path = os.path.join(folder_path, f"{sample}.txt")
#     with open(metadata_file_path, 'r') as f:
#         # Filter out lines that start with "experiment", "run", or are empty
#         metadata = "\n".join([line.strip() for line in f.readlines() if not line.startswith(("experiment", "run")) and line.strip() != ""])
#     return metadata
# 
# 
# # Filter rows where curated_biome doesn't match gpt_generated_biome
# mismatched_rows = m[m['curated_biome'] != m['gpt_generated_biome']].sort_values(by='curated_biome')
# 
# mismatched_rows = mismatched_rows[mismatched_rows['curated_biome'] == 'plant']
# 
# # Then proceed with your operations on the filtered dataframe
# 
# # Fetch and print metadata for each mismatched sample
# for index, row in mismatched_rows.iterrows():
#     metadata_for_sample = fetch_metadata_from_sample(row['sample'], directory_with_split_metadata)
#     print(f"Sample: {row['sample']}")
#     print(f"Curated Biome: {row['curated_biome']}")
#     print(f"GPT Generated Biome: {row['gpt_generated_biome']}")
#     print("Metadata:")
#     print(metadata_for_sample)
#     print("="*40)  # Separating line for readability
# =============================================================================



