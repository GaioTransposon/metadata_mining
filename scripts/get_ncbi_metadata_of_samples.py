#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:00:41 2023

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

path_to_sample_info = os.path.join(home, args.sample_info_biomes)
path_to_xml_files = os.path.join(home, args.xml_files)
path_to_output_dir = os.path.join(home, args.output_dir)


# # for testing purposes
# path_to_sample_info = os.path.join(home, "cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid_biome.csv")
# path_to_xml_files = os.path.join(home, "cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir")


# List of PMIDs
data = pd.read_csv(path_to_sample_info)

# Use str.split() method to split the string on each semicolon, expand=True will return DataFrame
# Then stack the DataFrame and reset the index to get a Series back
pmid_series = data['pmid_digits'].dropna().str.split(';', expand=True).stack().reset_index(drop=True)

# Convert Series to list after converting each entry to int
pmid_list = pmid_series.astype(int).tolist()

print(f"Number of PMIDs we want metadata about: {len(pmid_list)}")



# Create an empty dataframe
df = pd.DataFrame(columns=["pmid_digits", "title", "abstract"])

# Filter files
files = [f for f in os.listdir(path_to_xml_files) if f.endswith(".xml.gz")]
print(f"Number of files: {len(files)}")

# Process: 
no_files = 0
no_pmid_in_files = 0
no_found_pmid = 0

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
            pmid_element = article.find(".//PMID")
            if pmid_element is not None:
                pmid = int(pmid_element.text)  # Convert pmid to integer

                no_pmid_in_files+=1

                # If the PMID is in the list, extract the information
                if pmid in pmid_list:
                    print("PMID ", pmid, " found in list:")  # Print if we found a match

                    no_found_pmid+=1
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

                    # Add the data to the dataframe
                    df = pd.concat([df, pd.DataFrame({"pmid_digits": [str(pmid)], "title": [title], "abstract": [abstract]})], ignore_index=True)


print("total number of xml files processed over total: ", no_files, "/", len(files))
print("total number of pmids from all processed xml files ", no_pmid_in_files)
print("total number of pmids found in all files over total in our list", no_found_pmid, "/", len(pmid_list))



# merge data (biome, sample, pmid_digits) to this df (pmid, title, abstract):
data['pmid_digits'] = data['pmid_digits'].astype(str)
merged_df = pd.merge(data, df, on="pmid_digits", how="inner")  # inner gets only the common pmid_digits


# Save the dataframe to a CSV file
merged_df.to_csv(os.path.join(path_to_output_dir, 'sample_biome_pmid_title_abstract2.csv'), index=False)
print("Output file succesfully written")









