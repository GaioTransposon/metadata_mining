#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:56:39 2024

@author: dgaio
"""




import pandas as pd
import json
import re
import logging
import os
from datetime import datetime

# =======================================================
# PHASE 4: GPT Output Parsing
# =======================================================

class GPTOutputParsing:
    
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.unparsed_lines = []  # store lines that don't match the pattern
        self.parsed_data = None  # hold the df of parsed data

    def extract_contents_from_responses(self, gpt_responses):
        contents = []
        for response in gpt_responses:
            try:
                # access content in each response
                content = response['choices'][0]['message']['content']
                contents.append(content)  # collect each content block as a string
            except (KeyError, TypeError, IndexError) as e:
                logging.error(f"Error extracting content from response: {e}")
                contents.append("ERROR: Malformed response")
        return contents

    def parse_json_to_df(self, contents):
        data = []
        for content in contents:
            # remove any markdown code block syntax and split on triple backticks
            json_blocks = re.split(r'```json|```', content)
            for block in json_blocks:
                block = block.strip()
                if not block:
                    continue  # skip empty strings resulting from split
                if not block.startswith('{'):
                    block = '{' + block
                if not block.endswith('}'):
                    block += '}'
                try:
                    data_dict = json.loads(block)
                    data.append([
                        data_dict.get('sample-id', ''),
                        data_dict.get('biome-label', ''),
                        data_dict.get('geo-location', ''),
                        data_dict.get('keywords', ''),
                        data_dict.get('sub-biome', '')
                    ])
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON content: {e} -- {block[:50]}")
        self.parsed_data = pd.DataFrame(data, columns=['col_0', 'col_1', 'col_2', 'col_3', 'col_4'])

    def run(self, gpt_responses):
        contents = self.extract_contents_from_responses(gpt_responses)
        self.parse_json_to_df(contents)
        self.save_unparsed_to_file()
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



