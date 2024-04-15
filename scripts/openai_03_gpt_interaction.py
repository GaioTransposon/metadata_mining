#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:57:30 2023

@author: dgaio
"""


import os
import openai
import time
import logging
import glob 

# =======================================================
# PHASE 3: GPT Interaction
# =======================================================


class GPTInteractor:

    def __init__(self, work_dir, system_prompt_file, api_key_path, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, max_requests_per_minute):
        self.work_dir = work_dir
        self.system_prompt_file = system_prompt_file
        self.api_key = self.load_api_key(api_key_path)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_requests_per_minute = max_requests_per_minute
        
        self.request_times = []  # To track the timestamps of each request
        self.system_prompt = self.load_system_prompt()
        self.api_request_count = 0


    def load_api_key(self, api_key_path):
        try:
            with open(api_key_path, "r") as file:
                return file.read().strip()
        except Exception as e:
            logging.error(f"Error loading API key: {e}")
            return None
        
    def load_system_prompt(self):
        """Load the system prompt from a text file."""
        prompt_file = os.path.join(self.work_dir, self.system_prompt_file)
        print(prompt_file)
        try:
            with open(prompt_file, 'r') as file:
                return file.read().strip()
        except Exception as e:
            logging.error(f"Error loading system prompt: {e}")
            return None



    def load_latest_chunks_file(self):
        """Load the latest chunks file based on naming convention and timestamp."""
        file_pattern = os.path.join(self.work_dir, 'metadata_chunks_*.txt')
        list_of_files = glob.glob(file_pattern)  
        if not list_of_files:
            #print("No chunk files found.")
            return None
        latest_file = max(list_of_files, key=os.path.getctime)  
        print(latest_file)
        
        
        with open(latest_file, 'r') as file:
            content_strings = file.read().split("\n\n-----\n\n")
        # filter out any empty strings
        return [s for s in content_strings if s.strip()]
    

    def check_rate_limit(self):
        """Enforces the rate limit by sleeping if the request limit is reached."""
        current_time = time.time() 
        #
        # eventually…If problems with gpt errors → time.perf_counter() for more accurate timing, especially for short durations 
        #
        
        # Keep only the timestamps of requests made in the last minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # If we are about to exceed the rate limit, calculate the necessary wait time
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            logging.info(f"Approaching rate limit, waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            # Clean up the request times list after waiting
            self.request_times = [t for t in self.request_times if time.time() - t < 60]


        
    def gpt_request(self, content_string):
        
        self.check_rate_limit()
    
        if not self.system_prompt:
            logging.error("System prompt is not available. Aborting request.")
            return None
    
        # Count the number of samples in the content_string
        sample_count = content_string.count('sample_ID=')
    
        # Customize the system prompt based on the number of samples
        if sample_count == 1:
            customized_prompt = self.system_prompt.replace('microbial metagenomic samples', 'microbial metagenomic sample').replace('from their metadata texts', 'from its metadata text')
            #print('myprompt_1', customized_prompt)
        else:
            customized_prompt = self.system_prompt.replace('microbial metagenomic samples', f"{sample_count} microbial metagenomic samples")
            #print('myprompt_>1', customized_prompt)
    
        openai.api_key = self.api_key
    
        try:
            #print(content_string)
            # make the API request
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": customized_prompt},
                    {"role": "user", "content": content_string}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
    
            # Record the timestamp of this successful request
            self.request_times.append(time.time())
            self.api_request_count += 1
            print("####################")
            print('Api request count: ', self.api_request_count)
            print("####################")
    
            return response
    
        except openai.error.RateLimitError:
            logging.error("Rate limit exceeded.")
            return "RATE_LIMIT_EXCEEDED"
        except Exception as e:
            logging.error(f"GPT request failed: {e}")
            return None


  
    def get_gpt_responses(self):
        """Iterate over content strings and make requests to GPT."""
        content_strings = self.load_latest_chunks_file()
        if not content_strings:
            return []
    
        gpt_responses = []
        for content_string in content_strings:
            if not content_string.strip():
                continue
    
            response = self.gpt_request(content_string)
            
            if response == "RATE_LIMIT_EXCEEDED":
                logging.info("Rate limit exceeded. Waiting for 2 minutes...")
                print("Rate limit exceeded. Waiting for 1.5 minutes...")
                time.sleep(90)
            elif response is not None:
                gpt_responses.append(response)
    
        return gpt_responses
    
    
    def get_api_request_count(self):
        """
        Returns the current count of API requests made.
    
        Returns:
            int: The current API request count.
        """
        print("Current API request count:", self.api_request_count)
        return self.api_request_count





# =============================================================================
# test = GPTInteractor("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/", 
#                      "openai_system_prompt.txt",
#                      "/Users/dgaio/my_api_key",
#                      "gpt-3.5-turbo-1106",
#                      1.00,
#                      4096,
#                      0.75,
#                      0.25,
#                      1.5, 
#                      10000)
# 
# responses,api_request_count = test.get_gpt_responses()
# print(responses)
# print(api_request_count)
# =============================================================================




    