#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:14:56 2023

@author: dgaio
"""


import pandas as pd
import re
import logging


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
                    return file.read().splitlines()  
            except FileNotFoundError:
                logging.error(f"File '{self.filepath}' not found.")
                return None
            except IOError:
                logging.error(f"Error reading file '{self.filepath}'.")
                return None
        else:
            logging.error("No filepath provided.")
            return None

    def parse_samples(self):
        result = {}
        pattern = re.compile(r'(SRS|ERS|DRS)\d+__\w+__.*?__.*')
        
        for line in self.raw_content:
            match = pattern.match(line)
            if match:
                parts = line.split('__')
                if len(parts) == 4:
                    sample_id, biome, coordinates, location = parts
                    result[sample_id] = {
                        'biome': biome,
                        'geo_coordinates': coordinates,
                        'geo_text': location
                    }

        return result

    def prepare_dataframe(self, parsed_data_dict):
        return pd.DataFrame.from_dict(parsed_data_dict, orient='index').reset_index().rename(columns={'index': 'sample'})

    def save_cleaned_to_file(self):
        self.clean_filename = self.filepath.replace('gpt_raw_output', 'gpt_clean_output')
        self.parsed_data.to_csv(self.clean_filename, index=False)
        logging.info(f"Saved clean GPT output to: {self.clean_filename}")
        
    def run(self):
        parsed_samples = self.parse_samples()
        self.parsed_data = self.prepare_dataframe(parsed_samples)
        self.save_cleaned_to_file()
        return self.parsed_data
