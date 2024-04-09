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


# Logging function: logs to both the console and a file
def setup_logging():
    
    # determine the dir of the running script - logging file will be save there
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_filename = datetime.now().strftime("openai_validate_biomes_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(script_directory, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # captures all logs at the DEBUG level and above

    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    ###
    # sets up logging to file and to console with different log levels, but a common formatter.

    # file handler for logging
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # captures INFO level and higher logs (i.e., WARNING, ERROR, CRITICAL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # captures WARNING level and higher logs (i.e., ERROR, CRITICAL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    ###