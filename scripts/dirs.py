#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:27:39 2023

@author: dgaio
"""


# run as: 
# python dirs.py --input_file '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info' --output_dir '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs' --figure_path '~/cloudstor/Gaio/MicrobeAtlasProject/files_distribution_in_dirs.pdf'

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_directory_histogram(directory_path, figure_path=None):
    directory_counts = []

    for root, dirs, files in os.walk(directory_path):
        if os.path.basename(root).startswith('dir_'):
            directory_counts.append(len(files))

    bin_edges = np.arange(0, max(directory_counts) + 11, 10)
    bin_indices = np.digitize(directory_counts, bin_edges)

    bin_counts = np.bincount(bin_indices)

    nonzero_indices = np.nonzero(bin_counts)[0]
    first_nonzero_index = nonzero_indices[0] if len(nonzero_indices) > 0 else 0

    fig, ax = plt.subplots()
    ax.bar(range(first_nonzero_index, len(bin_counts)), bin_counts[first_nonzero_index:], align='center')
    ax.set_xlabel('File Count')
    ax.set_ylabel('Directory Count')
    ax.set_title('File Distribution Across Directories (Grouped by Bins of 10)')

    x_ticks = np.linspace(first_nonzero_index, len(bin_counts) - 1, num=4, dtype=int)
    x_tick_labels = [bin_edges[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    plt.tight_layout()

    if figure_path:
        plt.savefig(figure_path, format='pdf')

    plt.show()

parser = argparse.ArgumentParser(description='Split file into directories.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the split files')
parser.add_argument('--figure_path', type=str, default=None, help='Optional path to save the histogram figure')
args = parser.parse_args()

input_file = os.path.expanduser(args.input_file)
output_dir = os.path.expanduser(args.output_dir)
figure_path = os.path.expanduser(args.figure_path) if args.figure_path else None

start_time = time.time()

with open(input_file, 'r') as file:
    lines = file.readlines()

sample_text = ''
sample_count = 0
created_dirs = set()

for line in lines:
    if line != '\n':
        if line.startswith('>'):
            sample_name = line.replace('>', '').strip()
            last_3 = sample_name[-3:]
            dir_path = os.path.join(output_dir, 'dir_' + last_3)
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.add(dir_path)  # Track unique directories

            file_path = os.path.join(dir_path, sample_name + '.txt')
            sample_count += 1

            if sample_count % 100000 == 0:
                print(f"Processed {sample_count} samples...")

        sample_text += line
    else:
        if sample_text.endswith('\n'):
            with open(file_path, 'w') as output_file:
                output_file.write(sample_text)
        sample_text = ''

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print(f"Total directories created: {len(created_dirs)}")

plot_directory_histogram(output_dir, figure_path=figure_path)









