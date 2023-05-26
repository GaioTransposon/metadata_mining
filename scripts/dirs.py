#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:27:39 2023

@author: dgaio
"""

import os

# Path to the large file
large_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_10000'

# Directory to store the split files
output_directory = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)


# Open the large file for reading
with open(large_file_path, 'r') as file:
    lines = file.readlines()


# Iterate through each line in the file
sample_text = ''
for line in lines:
    if line != '\n':
        # Grab sample name
        if line.startswith('>'):
            sample_name = line.replace('>', '').strip()

            # Create directory based on last 3 digits of sample name
            last_3 = sample_name[-3:]
            dir_path = os.path.join(output_directory, 'dir_' + last_3)
            os.makedirs(dir_path, exist_ok=True)

            # Define file path for the sample
            file_path = os.path.join(dir_path, sample_name + '.txt')

        # Append the line to the sample text
        sample_text += line

    else:
        # Check if sample text ends with a blank line
        if sample_text.endswith('\n'):
            # Write the sample text to a file
            with open(file_path, 'w') as output_file:
                output_file.write(sample_text)

        # Reset the sample text
        sample_text = ''