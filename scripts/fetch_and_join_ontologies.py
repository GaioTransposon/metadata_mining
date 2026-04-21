#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# NB: code below is new (202604). Preevious version got ontology from Ontobee. Nowadays Ontobee not so reliable anymore. 

import argparse
import os
import pickle
import sys
import tempfile

import requests
import pronto


def print_error_box(message: str):
    """Prints a message inside a box made of hash characters."""
    width = len(message) + 4
    print("#" * width, "\n\n")
    print(f"# {message} #\n\n")
    print("#" * width)
    sys.stdout.flush()


def fetch_label_info(url: str, expected_prefix: str) -> dict:
    """
    Download an ontology file, parse it, and return a dictionary mapping
    ontology IDs (e.g. ENVO_00000001) to:
        'Term label (definition: ...)'
    """

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.Timeout:
        print_error_box(f"Timeout error for URL: {url}")
        return {}
    except requests.RequestException as e:
        print_error_box(f"Error fetching data from {url}: {str(e)}")
        return {}

    suffix = ".owl" if url.endswith(".owl") else ".obo"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        ontology = pronto.Ontology(tmp_path)
    except Exception as e:
        print_error_box(f"Error parsing ontology from {url}: {str(e)}")
        return {}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    label_info_dict = {}

    for term in ontology.terms():
        if term.obsolete:
            continue

        term_id = str(term.id).replace(":", "_")

        if not term_id.startswith(expected_prefix + "_"):
            continue

        if not term.name:
            continue

        definition = None
        if getattr(term, "definition", None):
            definition = str(term.definition).strip()

        if definition:
            joint_info = f"{term.name} (definition: {definition})"
        else:
            joint_info = term.name

        label_info_dict[term_id] = joint_info

    print(f"✔ Retrieved {len(label_info_dict)} entries from {expected_prefix}")

    return label_info_dict


def main(wanted_ontologies, output_dir, output_file):
    ontology_urls = {
        "FOODON": "http://purl.obolibrary.org/obo/foodon.owl",
        "ENVO": "http://purl.obolibrary.org/obo/envo.owl",
        "UBERON": "http://purl.obolibrary.org/obo/uberon.owl",
        "PO": "http://purl.obolibrary.org/obo/po.owl",
    }

    combined_dict = {}

    for ontology in wanted_ontologies:
        if ontology not in ontology_urls:
            print_error_box(f"Unknown ontology: {ontology}")
            continue

        url = ontology_urls[ontology]
        print(f"Fetching data from: {url}...")

        try:
            label_info_dict = fetch_label_info(url, ontology)
            combined_dict.update(label_info_dict)
        except Exception as e:
            print(f"Error fetching data from {url}: {str(e)}")

    combined_dict = dict(sorted(combined_dict.items()))

    expanded_output_dir = os.path.expanduser(output_dir)
    os.makedirs(expanded_output_dir, exist_ok=True)

    output_path = os.path.join(expanded_output_dir, output_file + ".pkl")

    with open(output_path, "wb") as f:
        pickle.dump(combined_dict, f)

    print(f"Dictionary saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and Join Ontologies")

    parser.add_argument(
        "--wanted_ontologies",
        nargs="+",
        help="List of wanted ontologies, separated by white space"
    )
    parser.add_argument(
        "--output_file",
        help="Name of the output dictionary without extension"
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save the output (default: current working dir)"
    )

    args = parser.parse_args()

    main(args.wanted_ontologies, args.output_dir, args.output_file)




















# Code below was used before 2026: 
# =============================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Oct 24 20:56:33 2023
# 
# @author: dgaio
# """
# 
# 
# 
# # run as: 
#     
# # python ~/github/metadata_mining/scripts/fetch_and_join_ontologies.py \
# #     --wanted_ontologies FOODON ENVO UBERON PO \
# #     --output_file ontologies_dict \
# #     --output_dir ~/MicrobeAtlasProject
# 
# # nb: # NCBITaxon on Ontobee is empty! 
# 
# 
# # Description: 
# # reads original ontologies
# # parses them, and outputs a dictionary of ontology terms in numeric format as keys 
# # their corresponding terms in text format as value. Then saves the dictionary. `
# 
# import argparse
# import requests
# import pandas as pd
# from io import StringIO
# import os
# import pickle
# import sys
# 
# 
# def print_error_box(message: str):
#     """Prints a message inside a box made of hash characters."""
#     width = len(message) + 4
#     print("#" * width, "\n\n")
#     print(f"# {message} #\n\n")
#     print("#" * width)
#     sys.stdout.flush()
# 
# 
# def fetch_label_info(url: str) -> dict:
#     """
#     Given a URL of a TSV file, this function downloads the data, processes it,
#     and returns a dictionary mapping labels to joint info.
# 
#     Args:
#         url (str): URL of the TSV file.
# 
#     Returns:
#         dict: Dictionary mapping labels to joint info.
#     """
#     
#     # Fetch the content from the URL
#     try:
#         response = requests.get(url, timeout=20)
#         response.raise_for_status()  # Raise an error if the request failed
#     except requests.Timeout:
#         print_error_box(f"Timeout error for URL: {url}")
#         return {}
#     except requests.RequestException as e:
#         print_error_box(f"Error fetching data from {url}: {str(e)}")
#         return {}
# 
#     # Use StringIO to simulate a file object
#     data = StringIO(response.text)
# 
#     # Read the TSV data into a pandas DataFrame
#     data = pd.read_csv(data, sep='\t')
#     
#     # Check if the DataFrame is empty
#     if data.empty:
#         print_error_box(f"No data returned from {url}. The DataFrame is empty.")
#         return {}
# 
#     # Extract ENVO labels from the 'Term IRI' and 'Parent term IRI' columns
#     data['label'] = data['Term IRI'].str.split('/').str[-1]
#     data['Parent_label'] = data['Parent term IRI'].str.split('/').str[-1]
# 
#     # Create a combined description column based on the presence of a definition
#     data['Joint_Info'] = data.apply(
#         lambda row: f"{row['Term label']} (definition: {row['Definition']})" if pd.notna(row['Definition']) else row['Term label'], 
#         axis=1
#     )
# 
#     # Create new DataFrames for the child and parent labels, text-labels, and other columns
#     child_df = data[['label', 'Term label', 'Definition', 'Joint_Info']]
#     parent_df = data[['Parent_label', 'Parent term label']].rename(columns={'Parent_label': 'label', 'Parent term label': 'Term label'}).assign(Joint_Info=data['Parent term label'])
# 
#     # Concatenate both DataFrames vertically
#     result = pd.concat([child_df, parent_df], axis=0, ignore_index=True)
# 
#     # Drop duplicates based on 'label' and 'Term label' columns
#     result = result.drop_duplicates(subset=['label', 'Term label'])
# 
#     # Keep only rows where the label starts with the specified patterns
#     print(f"Rows before filtering: {len(result)}")
#     print(result['label'].dropna().head(20).tolist())
#     print(result['label'].dropna().str.split('_').str[0].value_counts().head(20))
# 
#     patterns = ['ENVO_', 'NCBITaxon_', 'FOODON_', 'PO_', 'UBERON_']
#     mask = result['label'].str.startswith(tuple(patterns)).fillna(False)
#     result = result[mask]
# 
#     # Convert to dictionary
#     #old: label_info_dict = result.set_index('label')['Joint_Info'].to_dict()
#     label_info_dict = result.set_index('label')['Joint_Info'].to_dict()
# 
#     source_name = url.split("listTerms/")[-1].split("?")[0]
#     print(f"✔ Retrieved {len(label_info_dict)} entries from {source_name}")
# 
#     return label_info_dict
# 
# 
# 
# def main(wanted_ontologies, output_dir, output_file):
#     ontology_base_url = "https://ontobee.org/listTerms/{}?format=tsv"
#     ontology_urls = [ontology_base_url.format(ontology) for ontology in wanted_ontologies]
# 
#     # Initialize an empty dictionary to store combined results
#     combined_dict = {}
# 
#     for url in ontology_urls:
#         print(f"Fetching data from: {url}...")
#         try:
#             label_info_dict = fetch_label_info(url)
# 
#             # Merge the current dictionary into the combined dictionary
#             # This will overwrite duplicate keys with values from the current dictionary
#             combined_dict.update(label_info_dict)
#         
#         except Exception as e:
#             print(f"Error fetching data from {url}: {str(e)}")
# 
#     # Sort the combined dictionary by keys
#     combined_dict = dict(sorted(combined_dict.items()))
# 
#     # Save to the desired format (e.g., JSON)
#     expanded_output_dir = os.path.expanduser(output_dir)
#     os.makedirs(expanded_output_dir, exist_ok=True)  # <-- ONLY change
# 
#     output_path = os.path.join(expanded_output_dir, output_file + ".pkl")
# 
#     with open(output_path, 'wb') as f:
#         pickle.dump(combined_dict, f)
# 
#     print(f"Dictionary saved to {output_path}")
# 
# 
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Fetch and Join Ontologies')
#     parser.add_argument('--wanted_ontologies', nargs='+', help='List of wanted ontologies, separated by white space')
#     parser.add_argument('--output_file', help='Name of the output dictionary without extension')
# 
#     parser.add_argument(
#         '--output_dir',
#         default='.',                       # default = CWD
#         help='Directory to save the output (default: current working dir)'
#     )
# 
#     args = parser.parse_args()
# 
#     main(args.wanted_ontologies, args.output_dir, args.output_file)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# =============================================================================
