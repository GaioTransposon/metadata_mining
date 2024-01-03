#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:14:56 2023

@author: dgaio
"""


import pandas as pd
import re
import logging
import tiktoken


# =======================================================
# PHASE 3: GPT Output Parsing
# =======================================================


class GPTOutputParsing:
        
    def __init__(self, interactor_instance, encoding_name):
        self.filepath = interactor_instance.get_saved_filename()
        self.raw_content, self.raw_lines = self.load_from_file()
        self.parsed_data = None
        self.unparsed_lines = []  # Attribute to store unparsed lines
        self.clean_filename = None  
        self.unparsed_filename = None  # Filename for unparsed lines
        self.encoding_name = encoding_name
        
    def load_from_file(self):
        if self.filepath:
            try:
                with open(self.filepath, 'r') as file:
                    content = file.read()
                    lines = content.splitlines()
                    return content, lines
            except FileNotFoundError:
                logging.error(f"File '{self.filepath}' not found.")
                return None, None
            except IOError:
                logging.error(f"Error reading file '{self.filepath}'.")
                return None, None
        else:
            logging.error("No filepath provided.")
            return None, None
        
    def count_total_tokens(self, content):
        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(content)
        logging.info(f"Total output tokens: {len(tokens)}")

    def parse_samples(self):
        result = {}
        pattern = re.compile(r'^(SRS|ERS|DRS)\d+__\w+__.*')
        
        for line in self.raw_lines:
            # Check if the line is not empty
            if line.strip():  # .strip() removes whitespace from the beginning and end of the line
                match = pattern.match(line)
                if match:
                    parts = line.split('__')
                    sample_id = parts[0]
                    biome = parts[1]
                    location = parts[2] if len(parts) > 2 else None
                    sub_biome = parts[3] if len(parts) > 3 else None
                    additional_info = parts[4:] if len(parts) > 4 else None
        
                    result[sample_id] = {
                        'biome': biome,
                        'geo_text': location,
                        'sub_biome': sub_biome,
                        'additional_info': additional_info
                    }
                else:
                    self.unparsed_lines.append(line)  # Add to unparsed lines
        
        return result
    


    def save_unparsed_to_file(self):
        if self.unparsed_lines:
            self.unparsed_filename = self.filepath.replace('gpt_raw_output', 'gpt_unparsed_output')
            with open(self.unparsed_filename, 'w') as file:
                for line in self.unparsed_lines:
                    file.write(line + '\n')
            logging.info(f"Saved unparsed lines to: {self.unparsed_filename}")
        else:
            logging.info("No unparsed lines to save.")
            

    def prepare_dataframe(self, parsed_data_dict):
        return pd.DataFrame.from_dict(parsed_data_dict, orient='index').reset_index().rename(columns={'index': 'sample'})

    def save_cleaned_to_file(self):
        self.clean_filename = self.filepath.replace('gpt_raw_output', 'gpt_clean_output')
        self.parsed_data.to_csv(self.clean_filename, index=False)
        logging.info(f"Saved clean GPT output to: {self.clean_filename}")


    def run(self):
        self.count_total_tokens(self.raw_content)  
        parsed_samples = self.parse_samples()
        self.parsed_data = self.prepare_dataframe(parsed_samples)
        self.save_cleaned_to_file()
        self.save_unparsed_to_file()  # Save the unparsed lines
        return self.parsed_data







    


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














