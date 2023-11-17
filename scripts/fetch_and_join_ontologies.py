#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:56:33 2023

@author: dgaio
"""

# Description: 
# reads original ontologies
# parses them, and outputs a dictionary of ontology terms in numeric format as keys 
# their corresponding terms in text format as value. Then saves the dictionary. `

import argparse
import requests
import pandas as pd
from io import StringIO
import os
import pickle
import sys


def print_error_box(message: str):
    """Prints a message inside a box made of hash characters."""
    width = len(message) + 4
    print("#" * width, "\n\n")
    print(f"# {message} #\n\n")
    print("#" * width)
    sys.stdout.flush()


def fetch_label_info(url: str) -> dict:
    """
    Given a URL of a TSV file, this function downloads the data, processes it,
    and returns a dictionary mapping labels to joint info.

    Args:
        url (str): URL of the TSV file.

    Returns:
        dict: Dictionary mapping labels to joint info.
    """
    
    # Fetch the content from the URL
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()  # Raise an error if the request failed
    except requests.Timeout:
        print_error_box(f"Timeout error for URL: {url}")
        return {}
    except requests.RequestException as e:
        print_error_box(f"Error fetching data from {url}: {str(e)}")
        return {}

    # Use StringIO to simulate a file object
    data = StringIO(response.text)

    # Read the TSV data into a pandas DataFrame
    data = pd.read_csv(data, sep='\t')
    
    # Check if the DataFrame is empty
    if data.empty:
        print_error_box(f"No data returned from {url}. The DataFrame is empty.")
        return {}

    # Extract ENVO labels from the 'Term IRI' and 'Parent term IRI' columns
    data['label'] = data['Term IRI'].str.split('/').str[-1]
    data['Parent_label'] = data['Parent term IRI'].str.split('/').str[-1]

    # Create a combined description column based on the presence of a definition
    data['Joint_Info'] = data.apply(
        lambda row: f"{row['Term label']} (definition: {row['Definition']})" if pd.notna(row['Definition']) else row['Term label'], 
        axis=1
    )

    # Create new DataFrames for the child and parent labels, text-labels, and other columns
    child_df = data[['label', 'Term label', 'Definition', 'Joint_Info']]
    parent_df = data[['Parent_label', 'Parent term label']].rename(columns={'Parent_label': 'label', 'Parent term label': 'Term label'}).assign(Joint_Info=data['Parent term label'])

    # Concatenate both DataFrames vertically
    result = pd.concat([child_df, parent_df], axis=0, ignore_index=True)

    # Drop duplicates based on 'label' and 'Term label' columns
    result = result.drop_duplicates(subset=['label', 'Term label'])

    # Keep only rows where the label starts with the specified patterns
    patterns = ['ENVO_', 'NCBITaxon_', 'FOODON_', 'PO_', 'UBERON_']
    mask = result['label'].str.startswith(tuple(patterns)).fillna(False)
    result = result[mask]

    # Convert to dictionary
    label_info_dict = result.set_index('label')['Joint_Info'].to_dict()

    return label_info_dict


def main(wanted_ontologies, output_dir, output_file):
    ontology_base_url = "https://ontobee.org/listTerms/{}?format=tsv"
    ontology_urls = [ontology_base_url.format(ontology) for ontology in wanted_ontologies]

    # Initialize an empty dictionary to store combined results
    combined_dict = {}

    for url in ontology_urls:
        print(f"Fetching data from: {url}...")
        try:
            label_info_dict = fetch_label_info(url)

            # Merge the current dictionary into the combined dictionary
            # This will overwrite duplicate keys with values from the current dictionary
            combined_dict.update(label_info_dict)
        
        except Exception as e:
            print(f"Error fetching data from {url}: {str(e)}")

    # Sort the combined dictionary by keys
    combined_dict = dict(sorted(combined_dict.items()))

    # Save to the desired format (e.g., JSON)
    output_path = os.path.join(output_dir, output_file + ".pkl")

    with open(output_path, 'wb') as f:
        pickle.dump(combined_dict, f)


    print(f"Dictionary saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch and Join Ontologies')
    parser.add_argument('--wanted_ontologies', nargs='+', help='List of wanted ontologies, separated by white space')
    parser.add_argument('--output_dir', help='Directory to save the output')
    parser.add_argument('--output_file', help='Name of the output dictionary without extension')

    args = parser.parse_args()

    main(args.wanted_ontologies, args.output_dir, args.output_file)



# python /Users/dgaio/github/metadata_mining/scripts/fetch_and_join_ontologies.py \
#     --wanted_ontologies FOODON ENVO UBERON PO \    # NCBITaxon on Ontobee is empty! 
#     --output_dir "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject" \
#     --output_file "ontologies_dict"

# # on atlas
# python github/metadata_mining/scripts/fetch_and_join_ontologies.py --wanted_ontologies FOODON ENVO UBERON PO --output_dir "MicrobeAtlasProject" --output_file "ontologies_dict"
