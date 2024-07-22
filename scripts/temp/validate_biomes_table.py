#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:19:31 2024

@author: dgaio
"""


import os
import re
import pandas as pd
import pickle

def load_gold_standard(filepath):
    """ Load the gold standard dictionary from a pickle file. """
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def process_gpt_file(file_path, gold_dict_df):
    """ Process each GPT output file to calculate agreement with gold standard. """
    try:
        gpt_data = pd.read_csv(file_path, usecols=[0, 1], header=None, names=['sample', 'biome'])
        merged_df = gpt_data.merge(gold_dict_df, on='sample', suffixes=('_gpt', '_gold'))
        merged_df['agreement'] = merged_df['biome_gpt'] == merged_df['biome_gold']
        agreement_ratio = merged_df['agreement'].mean()
        valid_samples = merged_df['agreement'].count()  # Count only the matched samples
        return [agreement_ratio, valid_samples]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [None, 0]

def parse_filename(filename):
    """ Extract details from the filename using regex. """
    pattern = (
        r"gpt_clean_output_"
        r"(?P<nspb>nspb\d+)_"
        r"chunking(?P<chunking>yes|no)_"
        r"chunksize(?P<chunksize>\d+)_"
        r"model(?P<model>[\w.-]+)_"
        r"temp(?P<temp>\d\.\d)_"
        r"maxtokens(?P<maxtokens>\d+)_"
        r"topp(?P<topp>\d\.\d+)_"
        r"freqp(?P<freqp>\d\.\d+)_"
        r"presp(?P<presp>\d\.\d+)_"
        r"rs(?P<rs>\d+)_"
        r"(?P<miscellaneous>.*?)_"
        r"dt(?P<dt>\d{12})"
        r"(?:\.(txt|csv))"
    )
    match = re.match(pattern, filename)
    return match.groupdict() if match else None

def main():
    work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
    home_dir = "/Users/dgaio"
    gold_dict_path = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
    gold_dict = load_gold_standard(gold_dict_path)
    gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
    gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
    gold_dict_df.drop(columns='tuple_data', inplace=True)

    results_dict = {}
    gpt_files_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
    for filename in os.listdir(gpt_files_dir):
        if filename.startswith("gpt_clean_output_nspb"):
            file_path = os.path.join(gpt_files_dir, filename)
            results_dict[filename] = process_gpt_file(file_path, gold_dict_df)

    parsed_filenames = []
    for filename, results in results_dict.items():
        file_details = parse_filename(filename)
        if file_details:
            file_details.update({"filename": filename, "agreement_ratio": results[0], "sample_size": results[1]})
            parsed_filenames.append(file_details)

    results_df = pd.DataFrame(parsed_filenames)
    print(results_df.head())

    output_path = os.path.join(work_dir, 'gpt_outputs_biome_agreement.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
