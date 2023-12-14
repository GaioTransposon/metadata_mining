#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:56:48 2023

@author: dgaio
"""


import os
from datetime import datetime
import logging



# =======================================================
# PHASE 0: set up a logging system 
# =======================================================

# Define a logging function that logs to both the console and a file

class CustomFormatter(logging.Formatter):
    MAX_LENGTH = 250  # Set this to your desired length

    def format(self, record):
        if record.levelno == logging.DEBUG and len(record.msg) > self.MAX_LENGTH:
            record.msg = record.msg[:self.MAX_LENGTH] + "..."
        return super().format(record)
    

def setup_logging():
    # get dir of script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # give timestamp to log filename 
    log_filename = datetime.now().strftime("openai_validate_biomes_%Y%m%d_%H%M%S.log")
    
    # Join the directory with the log filename to get the full path
    log_filepath = os.path.join(script_directory, log_filename)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG to capture all types of logs

    formatter = CustomFormatter('%(asctime)s [%(levelname)s]: %(message)s')

    # File handler for logging
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # Set this to INFO to capture INFO and higher level logs
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Set this to WARNING to capture only WARNING and higher level logs
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)