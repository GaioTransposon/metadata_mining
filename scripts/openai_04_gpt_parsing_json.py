#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:56:39 2024

@author: dgaio
"""


import re
import pandas as pd
import logging
import os
from datetime import datetime


# =======================================================
# PHASE 4: GPT Output Parsing
# =======================================================



class GPTOutputParsing:
    
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.unparsed_lines = []  # Store lines that don't match the pattern
        self.parsed_data = None  # Hold the DataFrame of parsed data

    def parse_my_json_like(self, responses):
        
        responses = str(responses)
        print(type(responses))
        
        # Extract the content part only
        content_blocks = re.findall(r'"content":\s*"(.*?)",\s*"refusal"', responses, re.DOTALL)

        # Regex to capture everything from 'sample-id' to the next 'sample-id' or till the end
        pattern = re.compile(r'(\{\s*"sample-id".*?)(?=\{\s*"sample-id"|$)', re.DOTALL)
        
        data = []  # This will hold dictionaries for the DataFrame

        # Process each content block
        for block in content_blocks:
            # Unescape any escaped quotes or special characters (to deal with \n, \t, etc.)
            block = block.replace('\\n', '\n').replace('\\"', '"')

            # Find all blocks starting with 'sample-id'
            data_blocks = pattern.findall(block)

            # Process each data block and extract key-value pairs
            for data_block in data_blocks:
                # Use regex to capture key-value pairs in the block
                entry = {key.strip(): value.strip() for key, value in re.findall(r'"(.*?)":\s*"(.*?)"', data_block)}
                
                # Append the dictionary entry for each block
                if entry:
                    data.append(entry)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            print("DataFrame is empty.")
        else:
            # Rename columns to 'col_0', 'col_1', etc.
            df.columns = [f'col_{i}' for i in range(len(df.columns))]

        return df

    def run(self, gpt_responses):
        # Parse the content and store the DataFrame
        self.parsed_data = self.parse_my_json_like(gpt_responses)
        
        # Save any unparsed lines to a file
        #self.save_unparsed_to_file()
        
        # Return the parsed DataFrame
        return self.parsed_data

    def save_unparsed_to_file(self):
        if self.unparsed_lines:
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y%m%d%H%M')
            unparsed_file_path = os.path.join(self.work_dir, f"unparsed_{formatted_time}.txt")
            with open(unparsed_file_path, 'w') as file:
                for line in self.unparsed_lines:
                    file.write(line + '\n')
            logging.info("Saved unparsed lines.")
        else:
            logging.info("No unparsed lines to save.")




# =============================================================================
# ORIGINAL:
# 
# import pandas as pd
# import json
# import re
# import logging
# import os
# from datetime import datetime
# 
# # =======================================================
# # PHASE 4: GPT Output Parsing
# # =======================================================
# 
# class GPTOutputParsing:
#     
#     def __init__(self, work_dir):
#         self.work_dir = work_dir
#         self.unparsed_lines = []  # store lines that don't match the pattern
#         self.parsed_data = None  # hold the df of parsed data
# 
#     def extract_contents_from_responses(self, gpt_responses):
#         contents = []
#         for response in gpt_responses:
#             try:
#                 # access content in each response
#                 content = response['choices'][0]['message']['content']
#                 contents.append(content)  # collect each content block as a string
#             except (KeyError, TypeError, IndexError) as e:
#                 logging.error(f"Error extracting content from response: {e}")
#                 contents.append("ERROR: Malformed response")
#         return contents
# 
#     def parse_json_to_df(self, contents):
#         data = []
#         for content in contents:
#             # remove any markdown code block syntax and split on triple backticks
#             json_blocks = re.split(r'```json|```', content)
#             for block in json_blocks:
#                 block = block.strip()
#                 if not block:
#                     continue  # skip empty strings resulting from split
#                 if not block.startswith('{'):
#                     block = '{' + block
#                 if not block.endswith('}'):
#                     block += '}'
#                 try:
#                     data_dict = json.loads(block)
#                     data.append([
#                         data_dict.get('sample-id', ''),
#                         data_dict.get('biome-label', ''),
#                         data_dict.get('geo-location', ''),
#                         data_dict.get('keywords', ''),
#                         data_dict.get('sub-biome', '')
#                     ])
#                 except json.JSONDecodeError as e:
#                     logging.error(f"Failed to parse JSON content: {e} -- {block[:50]}")
#         self.parsed_data = pd.DataFrame(data, columns=['col_0', 'col_1', 'col_2', 'col_3', 'col_4'])
# 
#     def run(self, gpt_responses):
#         contents = self.extract_contents_from_responses(gpt_responses)
#         self.parse_json_to_df(contents)
#         self.save_unparsed_to_file()
#         return self.parsed_data
# 
#     def save_unparsed_to_file(self):
#         if self.unparsed_lines:
#             current_time = datetime.now()
#             formatted_time = current_time.strftime('%Y%m%d%H%M')
#             unparsed_file_path = os.path.join(self.work_dir, f"unparsed_{formatted_time}.txt")
#             with open(unparsed_file_path, 'w') as file:
#                 for line in self.unparsed_lines:
#                     file.write(line + '\n')
#             logging.info("Saved unparsed lines.")
#         else:
#             logging.info("No unparsed lines to save.")
# =============================================================================








# =============================================================================
# # WORKS FOR GPT_RAW 
# 
# 
# import re
# import pandas as pd
# import glob
# 
# # Define the base path and file pattern
# base_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/'
# file_pattern = 'gpt_raw_20241015*'
# file_paths = glob.glob(base_path + file_pattern)
# 
# 
# # List to collect DataFrames from each file
# data_frames = []
# 
# for file_path in file_paths:
#     # Read the contents of the file
#     with open(file_path, 'r') as file:
#         file_content = file.read()
# 
#     # Update the regex pattern to handle the last block
#     pattern = re.compile(r'\{(.*?"sample-id":.*?)(?=\{.*?"sample-id"|$)', re.DOTALL)
#     data_blocks = pattern.findall(file_content)
# 
#     # Prepare list to collect data
#     data = []
# 
#     # Extract data from each block
#     for block in data_blocks:
#         entry = {}
#         # Clean up the block
#         cleaned_block = re.sub(r'^.*?\{|\}.*$', '', block, flags=re.DOTALL)
#         # Extract key-value pairs
#         for line in cleaned_block.split('\n'):
#             key_value_match = re.search(r'"(.*?)":\s*"([^"]*)"', line)
#             if key_value_match:
#                 key, value = key_value_match.groups()
#                 entry[key.strip()] = value.strip()
#         if entry:
#             data.append(entry)
# 
#     # Convert list of dictionaries to DataFrame
#     df = pd.DataFrame(data)
# 
#     # Optionally rename columns to 'col_0', 'col_1', etc.
#     df.columns = [f'col_{i}' for i in range(len(df.columns))]
# 
#     # Append the DataFrame for this file to the list
#     data_frames.append(df)
# 
# # Concatenate all DataFrames into one
# final_df = pd.concat(data_frames, ignore_index=True)
# 
# # Display the final DataFrame
# print(final_df)
# =============================================================================




# =============================================================================
# 
# # Update the regex pattern to handle the last block
# # This pattern captures until it either finds another `{ "sample-id"` or reaches the end of the file.
# pattern = re.compile(r'\{(.*?"sample-id":.*?)(?=\{.*?"sample-id"|$)', re.DOTALL)
# data_blocks = pattern.findall(file_content)
# 
# # Prepare list to collect data
# data = []
# 
# # Extract data from each block
# for block in data_blocks:
#     entry = {}
#     # Remove trailing and leading characters outside the last closing brace
#     # This cleans up the content to ensure it's ready for key-value extraction.
#     cleaned_block = re.sub(r'^.*?\{|\}.*$', '', block, flags=re.DOTALL)
#     # Extract key-value pairs
#     for line in cleaned_block.split('\n'):
#         key_value_match = re.search(r'"(.*?)":\s*"([^"]*)"', line)
#         if key_value_match:
#             key, value = key_value_match.groups()
#             entry[key.strip()] = value.strip()
#     if entry:
#         data.append(entry)
# 
# # Convert list of dictionaries to DataFrame
# df = pd.DataFrame(data)
# 
# # Optionally rename columns to 'col_0', 'col_1', etc.
# df.columns = [f'col_{i}' for i in range(len(df.columns))]
# 
# # Display DataFrame
# print(df)
# =============================================================================








