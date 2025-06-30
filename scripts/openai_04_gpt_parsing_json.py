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

    
    def parse_my_json_like(self, content_list):
        """
        Parse a list of GPT 'content' strings, each in JSON-like format.
        """
        # Regex to capture everything from 'sample-id' to the next 'sample-id' or till the end
        pattern = re.compile(r'(\{\s*"sample-id".*?)(?=\{\s*"sample-id"|$)', re.DOTALL)
    
        data = []
    
        for content in content_list:
            # Clean up escape sequences
            content = content.replace('\\n', '\n').replace('\\"', '"')
    
            # Find all blocks starting with 'sample-id'
            data_blocks = pattern.findall(content)
    
            for block in data_blocks:
                entry = {key.strip(): value.strip()
                         for key, value in re.findall(r'"(.*?)":\s*"(.*?)"', block)}
                if entry:
                    data.append(entry)
    
        df = pd.DataFrame(data)
    
        if df.empty:
            print("DataFrame is empty.")
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
    
        return df
    

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
        # Extract content strings from response objects
        contents = [
            response.choices[0].message.content
            for response in gpt_responses
            if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content")
        ]
    
        # Now parse the JSON-like content
        self.parsed_data = self.parse_my_json_like(contents)
    
        # Save any unparsed lines (still useful)
        self.save_unparsed_to_file()
    
        return self.parsed_data






