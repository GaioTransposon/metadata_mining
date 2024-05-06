#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:14:56 2023

@author: dgaio
"""


import pandas as pd
import re
import logging
import os
import pickle
from datetime import datetime
import re


# =======================================================
# PHASE 4: GPT Output Parsing
# =======================================================


class GPTOutputParsing:
        
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.unparsed_lines = []  # Store lines that don't match the pattern
        self.parsed_data = None  # Will hold the DataFrame of parsed data
        

    
    def extract_contents_from_responses(self, gpt_responses):
        contents = []
        for response in gpt_responses:
            try:
                # Accessing the content of the first choice in each response
                content = response.choices[0].message['content']
                contents.extend(content.split('\n'))  # Splitting by newline to get individual lines
            except (KeyError, AttributeError, IndexError) as e:
                logging.error(f"Error extracting content from response: {e}")
                contents.append("ERROR: Malformed response")

        return contents



# =============================================================================
#     def parse_responses(self, gpt_responses):
#         responses_content = self.extract_contents_from_responses(gpt_responses)
#         sample_pattern = re.compile(r'(ERS|SRS|DRS)\d+__.*')  
# 
#         parsed_samples = []
#         for line in responses_content:
#             match = sample_pattern.search(line)
#             if match:
#                 matched_sample = match.group()
#                 parts = matched_sample.split('___')
#                 sample_dict = {f'col_{i}': part for i, part in enumerate(parts)}
#                 parsed_samples.append(sample_dict)
#             else:
#                 if line.strip():
#                     self.unparsed_lines.append(line)
# 
#         self.parsed_data = pd.DataFrame(parsed_samples)
# =============================================================================

    def parse_responses(self, gpt_responses):
        responses_content = self.extract_contents_from_responses(gpt_responses)
        # pattern that captures sequences separated by two to four underscores
        sample_pattern = re.compile(r'(ERS|SRS|DRS)\d+(_{2,4}).*')
    
        parsed_samples = []
        for line in responses_content:
            match = sample_pattern.search(line)
            if match:
                matched_sample = match.group()
                # Splitting the line based on two to four underscores
                parts = re.split(r'_{2,4}', matched_sample)
                sample_dict = {f'col_{i}': part for i, part in enumerate(parts)}
                parsed_samples.append(sample_dict)
            else:
                if line.strip():
                    self.unparsed_lines.append(line)
    
        self.parsed_data = pd.DataFrame(parsed_samples)


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

    
    
    def run(self, gpt_responses):
        self.parse_responses(gpt_responses)
        self.save_unparsed_to_file()

        return self.parsed_data



# =============================================================================
# parser = GPTOutputParsing("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/")
# parsed_data = parser.run(responses)
# 
# print(parsed_data)
# =============================================================================





# =============================================================================
# # Replace 'path_to_your_file.txt' with the path to your raw file
# myfile = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_raw_output_nspb200_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20231221_1420.txt'
# 
# import re
# 
# def parse_file(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()
# 
#     # Replace 1), 2), 3) with '__'
#     content = re.sub(r'\d+\)', '__', content)
# 
#     # Replace newline characters with '__'
#     content = content.replace('\n', '__')
# 
#     # Replace tab characters with '__'
#     content = content.replace('\t', '__')
# 
#     # Replace multiple underscores with a double underscore
#     content = re.sub(r'__+', '__', content)
# 
#     # Replace single underscore with a double underscore
#     content = re.sub(r'_+', '__', content)
# 
#     # Split the content at each occurrence of SRS, DRS, or ERS
#     records = re.split(r'(?=SRS|DRS|ERS)', content)
# 
#     # Filter out items that don't start with SRS, DRS, or ERS
#     records = [record for record in records if record.startswith(('SRS', 'DRS', 'ERS'))]
# 
#     # Strip each item, remove empty items, and join into a single string
#     processed_records = ['__'.join(filter(None, [item.strip() for item in record.split('__')])) for record in records]
# 
#     return processed_records
# 
# # Replace 'path_to_your_file.txt' with the path to your raw file
# processed_records = parse_file(myfile)
# print(f"Number of processed records: {len(processed_records)}")
# for record in processed_records[:10]:  # Print first 10 records for preview
#     print(record)
# 
# 
# len(processed_records)
# 
# =============================================================================