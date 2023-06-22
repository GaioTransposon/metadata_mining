#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:27:39 2023

@author: dgaio
"""

import os
import time

# Path to the large file
large_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info'

# Directory to store the split files
output_directory = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Start the timer
start_time = time.time()

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

# Calculate and print the elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# it's going to take 26 minutes for the whole sample.info, in fact: 
# python ./github/metadata_mining/scripts/dirs.py
# Elapsed time: 1503.12704205513 second




# Have a look at the distribution of files across directories: 
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_directory_histogram(directory_path):
    directory_counts = []

    for root, dirs, files in os.walk(directory_path):
        if os.path.basename(root).startswith('dir_'):
            directory_counts.append(len(files))

    # Group directory counts into bins of 10
    bin_edges = np.arange(0, max(directory_counts) + 11, 10)
    bin_indices = np.digitize(directory_counts, bin_edges)

    # Calculate bin counts
    bin_counts = np.bincount(bin_indices)

    # Find the first non-zero bin count index
    nonzero_indices = np.nonzero(bin_counts)[0]
    first_nonzero_index = nonzero_indices[0] if len(nonzero_indices) > 0 else 0

    # Plot the histogram
    fig, ax = plt.subplots()
    ax.bar(range(first_nonzero_index, len(bin_counts)), bin_counts[first_nonzero_index:], align='center')
    ax.set_xlabel('File Count')
    ax.set_ylabel('Directory Count')
    ax.set_title('File Distribution Across Directories (Grouped by Bins of 10)')

    # Set x-axis tick positions and labels
    x_ticks = np.linspace(first_nonzero_index, len(bin_counts) - 1, num=4, dtype=int)
    x_tick_labels = [bin_edges[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    plt.tight_layout()
    plt.show()

    
    
# Specify the path to the parent directory containing subdirectories with files
parent_directory = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs'

# Call the function to plot the directory histogram
plot_directory_histogram(parent_directory)





