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
        
    def __init__(self, interactor_instance, encoding_name, processed_sample_ids):
        self.filepath = interactor_instance.get_saved_filename()
        self.raw_content, self.raw_lines = self.load_from_file()
        self.parsed_data = None
        self.unparsed_lines = []  # Attribute to store unparsed lines
        self.clean_filename = None  
        self.unparsed_filename = None  # Filename for unparsed lines
        self.encoding_name = encoding_name
        self.processed_sample_ids = processed_sample_ids 
        self.interactor_instance = interactor_instance
        
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
        result = []
        pattern = re.compile(r'^(ERS|SRS|DRS)\d+__.*')
        
        for line in self.raw_lines:
            # Check if the line is not empty and matches the pattern
            if line.strip() and pattern.match(line.strip()):
                parts = line.split('__')
                sample_dict = {}
                for i, part in enumerate(parts):
                    sample_dict[f'col_{i}'] = part
                result.append(sample_dict)
            else:
                if line.strip():  # Add non-empty lines that do not match the pattern to unparsed lines
                    self.unparsed_lines.append(line)
        
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
            

    def prepare_dataframe(self, parsed_data_list):
        return pd.DataFrame(parsed_data_list)


    def save_cleaned_to_file(self):
        self.clean_filename = self.filepath.replace('gpt_raw_output', 'gpt_clean_output')
        self.parsed_data.to_csv(self.clean_filename, index=False)
        logging.info(f"Saved clean GPT output to: {self.clean_filename}")



    def parse_direct_responses(self, gpt_responses):
        contents = []
        for response in gpt_responses:
            try:
                content = response.choices[0].message['content']
                contents.append(content)
            except KeyError:
                logging.error("Malformed GPT response")
    
        # Convert the list of contents to a '\n' separated string, mimicking the file content
        self.raw_content = '\n'.join(contents)
        self.raw_lines = self.raw_content.splitlines()
    
        # Now you can reuse the existing parsing logic
        parsed_samples = self.parse_samples()
        self.parsed_data = self.prepare_dataframe(parsed_samples)
        self.save_unparsed_to_file()
        self.save_cleaned_to_file()
    
        return self.parsed_data
    

    def run(self):
        # Existing logic to count tokens, parse samples, prepare DataFrame, and save files
        self.count_total_tokens(self.raw_content)
        parsed_samples = self.parse_samples()
        self.parsed_data = self.prepare_dataframe(parsed_samples)
        self.save_cleaned_to_file()
        self.save_unparsed_to_file()  # Save the unparsed lines
    
        # New logic to find and log missing sample IDs
        # Assuming 'col_0' is where your sample IDs are stored after parsing. Adjust if necessary.
        parsed_sample_ids = set(self.parsed_data['col_0'].unique())
        missing_sample_ids = set(self.processed_sample_ids) - parsed_sample_ids
    
        if missing_sample_ids:
            missing_samples_str = ', '.join(missing_sample_ids)
            logging.warning(f"Missing sample IDs not present in parsed DataFrame: {missing_samples_str}")
            print(f"Missing sample IDs: {missing_samples_str}")
        else:
            logging.info("No missing sample IDs. All processed samples are present in parsed DataFrame.")
            print("No missing sample IDs.")
    
        # Return the parsed DataFrame as before
        return self.parsed_data, missing_sample_ids



    


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