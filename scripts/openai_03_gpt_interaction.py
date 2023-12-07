#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:57:30 2023

@author: dgaio
"""


import os
import openai
from datetime import datetime
import time
import logging
import tiktoken


# =======================================================
# PHASE 2: GPT Interaction
# =======================================================

class GPTInteractor:
   
    def __init__(self, work_dir, n_samples_per_biome, chunk_size, system_prompt_file, api_key_path, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
        self.work_dir = work_dir
        self.n_samples_per_biome = n_samples_per_biome
        self.chunk_size = chunk_size
        self.system_prompt_file = system_prompt_file
        self.api_key_path = api_key_path
        self.api_key = self.load_api_key()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.saved_filename = None  # This will store the filename once saved


    def token_count(self, text, encoding_name="cl100k_base"):
        """Return the number of tokens in the text using tiktoken."""
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    
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
            #print(chunk)
            print("----")
        
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

        
    def load_system_prompt(self):
        """Load the system prompt from a text file."""
        prompt_file = os.path.join(self.work_dir, self.system_prompt_file)
        try:
            with open(prompt_file, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"System prompt file '{prompt_file}' not found.")
            return None
        except IOError:
            print(f"Error reading system prompt file '{prompt_file}'.")
            return None
        
    def gpt_request(self, content_string):
        openai.api_key = self.api_key
        system_prompt = self.load_system_prompt()  # Load the system prompt

        if not system_prompt:
            print("System prompt is not available. Aborting request.")
            return None

        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt  # Use the loaded system prompt
                },
                {
                    "role": "user",
                    "content": content_string
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
        - List of responses from GPT.
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
        self.saved_filename = f"gpt_raw_output_nspb{self.n_samples_per_biome}_chunksize{self.chunk_size}_model{self.model}_temp{self.temperature}_maxtokens{self.max_tokens}_topp{self.top_p}_freqp{self.frequency_penalty}_presp{self.presence_penalty}_dt{current_datetime}.txt"
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
    



