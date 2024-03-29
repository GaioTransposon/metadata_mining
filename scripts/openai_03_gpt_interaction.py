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
import glob 
import json


# =======================================================
# PHASE 2: GPT Interaction
# =======================================================

class GPTInteractor:

    def __init__(self, work_dir, n_samples_per_biome, chunk_size, seed, system_prompt_file, api_key_path, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, opt_text):
        self.work_dir = work_dir
        self.n_samples_per_biome = n_samples_per_biome
        self.chunk_size = chunk_size 
        self.seed = seed
        self.system_prompt_file = system_prompt_file
        self.api_key_path = api_key_path
        self.api_key = self.load_api_key()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.opt_text = opt_text
        self.saved_filename = None
        self.api_request_count = 0

    

    def load_latest_chunks_file(self):
        """Load the latest chunks file based on naming convention and timestamp."""
        file_pattern = os.path.join(self.work_dir, 'metadata_chunks_*.txt')
        list_of_files = glob.glob(file_pattern)  
        if not list_of_files:
            #print("No chunk files found.")
            return None
        latest_file = max(list_of_files, key=os.path.getctime)  # get the latest file
        with open(latest_file, 'r') as file:
            content_strings = file.read().split("\n\n-----\n\n")
        
        # filter out any empty strings
        return [s for s in content_strings if s.strip()]

    
    def load_api_key(self):
        try:
            with open(self.api_key_path, "r") as file:
                api_key = file.read().strip()
                return api_key.strip()
        except FileNotFoundError:
            logging.error(f"File '{self.api_key_path}' not found.")
            return None
        except IOError:
            logging.error(f"Error reading file '{self.api_key_path}'.")
            return None

        
    def load_system_prompt(self):
        """Load the system prompt from a text file."""
        prompt_file = os.path.join(self.work_dir, self.system_prompt_file)
        try:
            with open(prompt_file, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            logging.error(f"System prompt file '{prompt_file}' not found.")
            return None
        except IOError:
            logging.error(f"Error reading system prompt file '{prompt_file}'.")
            return None
        
    
    def get_api_request_count(self):
        """Returns the total number of API requests made."""
        return self.api_request_count


    
    
    
    def gpt_request(self, content_string):
        # Set the API key only once, preferably in the __init__ method or right before making a request, but not with every request
        openai.api_key = self.api_key
    
        # Load the system prompt first to ensure it's available before making the request
        system_prompt = self.load_system_prompt()
        if not system_prompt:
            logging.error("System prompt is not available. Aborting request.")
            return None
    
        # Make the API request
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt  # use the loaded system prompt
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
    
        
        #print(response)
        # Increment the API request counter after a successful request
        self.api_request_count += 1
    
        return response
    
        
   

    def interact_with_gpt(self, specific_chunks=None):
        """Iterate over content strings and make requests to GPT. Accepts an optional list of specific chunks."""
        # Use the provided specific_chunks if any, otherwise load the latest chunks file
        
        content_strings = specific_chunks if specific_chunks is not None else self.load_latest_chunks_file()
        if not content_strings:
            return []
    
        gpt_responses = []
    
        for content_string in content_strings:
            if not content_string.strip():  # skip empty content strings
                continue
    
            # Send request to API
            try:
                # print(content_string)
                response = self.gpt_request(content_string=content_string)
                #print(response)
                gpt_responses.append(response)
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
        # extract the "content" from each response with error handling
        contents = []
        for response in gpt_responses:
            try:
                contents.append(response['choices'][0]['message']['content'])
            except KeyError:
                contents.append("ERROR: Malformed response")
        
        # join all contents with a separator (two newlines for readability)
        final_content = "\n\n".join(contents)
        
        # construct the filename
        api_count = self.get_api_request_count()
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
        self.saved_filename = f"gpt_raw_output_nspb{self.n_samples_per_biome}_chunksize{self.chunk_size}_model{self.model}_temp{self.temperature}_maxtokens{self.max_tokens}_topp{self.top_p}_freqp{self.frequency_penalty}_presp{self.presence_penalty}_rs{self.seed}_API{api_count}_{self.opt_text}_dt{current_datetime}.txt"
        #self.saved_filename = f"gpt_raw_output_nspb{self.n_samples_per_biome}_chunksize{self.chunk_size}_model{self.model}_temp{self.temperature}_maxtokens{self.max_tokens}_topp{self.top_p}_freqp{self.frequency_penalty}_presp{self.presence_penalty}_dt{current_datetime}.txt"
        self.saved_filename = os.path.join(self.work_dir, self.saved_filename)
    
        # write to file
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
            logging.error("No file has been saved yet!")
            return None
    
    
    def run(self):
        print("Starting interaction with GPT...")
        gpt_responses = self.interact_with_gpt()
        print("Finished interaction with GPT.")
        self.save_gpt_responses_to_file(gpt_responses)
    





    