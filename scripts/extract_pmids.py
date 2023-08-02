#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:10:34 2023

@author: dgaio
"""

# run as: 
# python extract_pmids.py --dir '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs' --output_csv '~/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid.csv' --plot --figure_path '~/cloudstor/Gaio/MicrobeAtlasProject/pmids_NaN_vs_nonNaN.pdf'


import os
import re
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def find_pmids(directory):
    pmid_dict = {}
    pmid_pattern = re.compile(r"PMID[^0-9]*\d{8}", re.IGNORECASE)

    file_counter = 0
    start_time = time.time()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    if lines:
                        key = lines[0].strip().replace(">","")
                        pmid_dict[key] = pmid_dict.get(key, [])
                        for line in lines[1:]:
                            match = pmid_pattern.search(line)
                            if match:
                                pmid_dict[key].append(match.group())
                
                file_counter += 1
                
                if file_counter % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Processed {file_counter} files in {elapsed_time:.2f} seconds "
                          f"({file_counter / elapsed_time:.2f} files/second)")

    return pmid_dict


def dict_to_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample', 'pmid', 'pmid_digits'])

        for key, values in dictionary.items():
            if values:
                pmid = '&'.join(values)
                pmid_digits = re.findall(r'\d+', pmid)
                first_digits = pmid_digits[0] if pmid_digits else ''
                writer.writerow([key, pmid, first_digits])
            else:
                writer.writerow([key, '', ''])

                
def plot_pmid_info(filename, figure_path=None):
    data = pd.read_csv(filename)
    non_na_count = data['pmid_digits'].notna().sum()
    na_count = data['pmid_digits'].isna().sum()

    labels = ['Non-NA', 'NA']
    counts = [non_na_count, na_count]

    plt.bar(labels, counts)
    plt.xlabel('Column "pmid_digits"')
    plt.ylabel('Count')
    plt.title('Count of Non-NA Values in Column "pmid_digits"')

    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    if figure_path:
        plt.savefig(figure_path, format='pdf')
    else:
        plt.show()


parser = argparse.ArgumentParser(description='Find PMIDs in directory.')
parser.add_argument('--dir', type=str, required=True, help='Directory to search for PMIDs')
parser.add_argument('--output_csv', type=str, required=True, help='Output csv file')
parser.add_argument('--plot', action='store_true', help='Plot the count of non-NA values')
parser.add_argument('--figure_path', type=str, help='Optional path to save the histogram figure')
args = parser.parse_args()

dir_path = os.path.expanduser(args.dir)
output_csv = os.path.expanduser(args.output_csv)
figure_path = os.path.expanduser(args.figure_path) if args.figure_path else None



pmid_dict = find_pmids(dir_path)
dict_to_csv(pmid_dict, output_csv)

if args.plot:
    plot_pmid_info(output_csv, figure_path=figure_path)



