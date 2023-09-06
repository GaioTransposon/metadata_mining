#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:55:43 2023

@author: dgaio
"""



GOLD_DICT_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.json"
CSV_PATH = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv"


import pandas as pd
import json

def load_gold_dict(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_gold_dict(gold_dict, filename):
    with open(filename, 'w') as f:
        json.dump(gold_dict, f)


def get_sub_df_with_lowest_count(df, gold_dict, iteration):
    min_count = float('inf')
    min_biome = None
    for biome in df['biome'].unique():
        if biome == 'unknown' and iteration % 10 != 0:  # Skip 'unknown' most of the time (this is because most unknowns turn out to be animal)
            continue
        if biome not in gold_dict:
            gold_dict[biome] = []
        count = len(gold_dict[biome])
        if count < min_count:
            min_count = count
            min_biome = biome
    sub_df = df[df['biome'] == min_biome].sort_values(by='biome_counts', ascending=False)
    return sub_df


def print_gold_dict_stats(gold_dict):
    print("Gold dictionary statistics:")
    for biome, pmids in gold_dict.items():
        print(f"{biome}: {len(pmids)}")
        

def main():
    filename = CSV_PATH
    df = pd.read_csv(filename)
    
    # Group by 'pmid' and 'biome' and count the rows for each group
    grouped_df = df.groupby(['pmid', 'biome']).size().reset_index(name='biome_counts')

    # Optionally, merge this information back into the original DataFrame, if needed.
    merged_df = pd.merge(df[['pmid', 'title', 'abstract']].drop_duplicates(), grouped_df, on='pmid')

    
    # Replace 'aquatic' with 'water' in the 'biome' column
    merged_df['biome'] = merged_df['biome'].replace('aquatic', 'water')

    # Sort by title and (evt) take the first n rows
    df = merged_df.sort_values(by='title')         #.head(20)
    
    gold_dict_filename = GOLD_DICT_PATH
    gold_dict = load_gold_dict(gold_dict_filename)
    
    # Remove PMIDs already in gold_dict from DataFrame
    already_processed_pmids = [pmid for pmid_list in gold_dict.values() for pmid in pmid_list]
    df = df[~df['pmid'].isin(already_processed_pmids)]

    iteration = 0  # Initialize an iteration counter
    while True:
        iteration += 1  # Increment the iteration counter
        sub_df = get_sub_df_with_lowest_count(df, gold_dict, iteration)
        
        if sub_df.empty:
            print("No more records to process for the least populated biome. Exiting.")
            print("Current gold_dict content:")
            print(gold_dict)
            print_gold_dict_stats(gold_dict)
            break

        row = sub_df.iloc[0]

        question = f"\n\n\nIs the biome: {row['biome']}?"
        print(f"\nTitle: {row['title']}")
        print(f"\nAbstract: {row['abstract']}\n\n")
        answer = input(question + " (y/n/q): ")

        if answer == 'q':
            print("Current gold_dict content:")
            print(gold_dict)
            print_gold_dict_stats(gold_dict)
            break
        elif answer == 'y':
            gold_dict[row['biome']].append(row['pmid'])
            df = df[df['pmid'] != row['pmid']]
        else:
            new_biome = input("Which biome is it? (a for animal, w for water, s for soil, p for plant, u for unknown): ")
            biome_map = {'a': 'animal', 'w': 'water', 's': 'soil', 'p': 'plant', 'u': 'unknown'}
            new_biome_str = biome_map[new_biome]
            if new_biome_str not in gold_dict:
                gold_dict[new_biome_str] = []
            gold_dict[new_biome_str].append(row['pmid'])
            df = df[df['pmid'] != row['pmid']]

    save_gold_dict(gold_dict, gold_dict_filename)


if __name__ == "__main__":
    main()








        
    

