#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:35:51 2023

@author: dgaio
"""


# run as: 
# python github/metadata_mining/scripts/get_ncbi_metadata_of_samples.py --sample_info_biomes "cloudstor/Gaio/MicrobeAtlasProject/samples_biomes.csv" --xml_files "cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir" --output_dir "cloudstor/Gaio/MicrobeAtlasProject/"

import pandas as pd
import os
import gzip
from lxml import etree
from tqdm import tqdm
import argparse  



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process XML files.')
parser.add_argument('--sample_info_biomes', type=str, required=True, help='Path to the sample info file.')
parser.add_argument('--xml_files', type=str, required=True, help='Path to the directory with XML files.')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the results.')
args = parser.parse_args()

home = os.path.expanduser('~')

path_to_sample_info = os.path.join(home, args.sample_info)
path_to_xml_files = os.path.join(home, args.xml_files)
path_to_output_dir = os.path.join(home, args.output_dir)


# # for testing purposes
# path_to_sample_info = os.path.join(home, "cloudstor/Gaio/MicrobeAtlasProject/samples_biomes.csv")
# path_to_xml_files = os.path.join(home, "cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir")

# List of Sample Numbers
data = pd.read_csv(path_to_sample_info)
sample_list = data['sample'].dropna().tolist()
print(f"Number of Sample Numbers we want metadata about: {len(sample_list)}")

# Create an empty dataframe
df = pd.DataFrame(columns=["sample", "title", "abstract", "pmid"])

# Filter files
files = [f for f in os.listdir(path_to_xml_files) if f.endswith(".xml.gz")]
print(f"Number of files: {len(files)}")

# Process: 
no_files = 0
no_sample_in_files = 0
no_found_sample = 0

# Go through each file
for filename in tqdm(files, desc="Processing files"):
    
    no_files+=1 
    print("\nProcessing file number: ", no_files,'/',len(files))
    
    # Generate full file path
    filepath = os.path.join(path_to_xml_files, filename)
    with gzip.open(filepath, 'rt') as f:  # 'rt' mode is for reading text data.
        tree = etree.parse(f)
        root = tree.getroot()

        # Go through each article
        for article in tqdm(root.findall(".//PubmedArticle"), desc=f"Processing {filename}", leave=False):
            accession_number_elements = article.findall(".//AccessionNumber")
            for accession_number_element in accession_number_elements:
                sample = accession_number_element.text  # Accession number is a string

                if sample is not None and sample.startswith(("DRS", "ERS", "SRS")):  # Check if sample starts with "DRS", "ERS", or "SRS"
                    no_sample_in_files+=1
                    print('looking for sample: ', sample)

                    # If the sample is in the list, extract the information
                    if sample in sample_list:
                        print("Sample ", sample, " found in list:")  # Print if we found a match

                        no_found_sample+=1
                        title_element = article.find(".//ArticleTitle")
                        if title_element is not None:
                            title = title_element.text
                        else:
                            title = None

                        abstract_element = article.find(".//AbstractText")
                        if abstract_element is not None:
                            abstract = abstract_element.text
                        else:
                            abstract = None

                        pmid_element = article.find(".//PMID")
                        if pmid_element is not None:
                            pmid = pmid_element.text  # PMID is a string
                        else:
                            pmid = None

                        # Add the data to the dataframe
                        df = pd.concat([df, pd.DataFrame({"sample": [sample], "title": [title], "abstract": [abstract], "pmid": [pmid]})], ignore_index=True)






print("total number of xml files processed over total: ", no_files, "/", len(files))
print("total number of samples from all processed xml files ", no_sample_in_files)
print("total number of samples found in all files over total in our list", no_found_sample, "/", len(sample_list))


# Save the dataframe to a CSV file
df.to_csv(os.path.join(path_to_output_dir, 'sample_biome_pmid_title_abstract.csv'), index=False)
print("Output file succesfully written")


