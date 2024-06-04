#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:14:25 2024

@author: dgaio
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

def load_and_filter_data(csv_path, filter_conditions):
    """ Load CSV and filter based on conditions provided in a dictionary. """
    data = pd.read_csv(csv_path)
    for key, value in filter_conditions.items():
        data = data[data[key] == value]
    return data

def identify_differing_columns(data):
    """ Identify columns that have more than one unique value in the dataframe. """
    differing_cols = [col for col in data.columns if len(data[col].unique()) > 1]
    return differing_cols

def plot_agreement(data):
    """ Plot agreement ratios as a bar plot with sample sizes annotated. """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data['filename'], data['agreement_ratio'], color='skyblue')
    plt.xlabel('File Name')
    plt.ylabel('Agreement Ratio')
    plt.title('Agreement Ratio by File with Sample Size')
    plt.xticks(rotation=45, ha='right')

    # Annotate sample sizes on bars
    for bar, sample_size in zip(bars, data['sample_size']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(sample_size)}', va='bottom')  # va: vertical alignment

    plt.tight_layout()
    plt.show()

def main(args):
    # Assume the CSV file path is fixed or passed as the first argument
    csv_path = args[0]
    filter_conditions = dict(pair.split('=') for pair in args[1:])
    
    data = load_and_filter_data(csv_path, filter_conditions)
    if data.empty:
        print("No data found for the given filters.")
        return

    print("These are the files selected:", data['filename'].tolist())
    
    differing_cols = identify_differing_columns(data)
    if differing_cols:
        print("Differing features among selected files:", differing_cols)
    else:
        print("No differing features among selected files.")
    
    plot_agreement(data)

if __name__ == "__main__":
    main(sys.argv[1:])
