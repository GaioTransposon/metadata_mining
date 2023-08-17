#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:00:41 2023

@author: dgaio
"""


# conda install xmltodict
# conda install -c conda-forge pygraphviz
# conda install -c conda-forge networkx


# run as: 
python ~/github/metadata_mining/scripts/get_ncbi_metadata.py  \
        --work_dir '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/' \
            --xml_files_dir "ncbi_metadata_dir" \
                --unique_pmids "unique_pmids" \
                    --unique_pmcids "unique_pmcids" \
                        --output_file "ncbi_metadata_selection"
    
 

import pandas as pd
import os
import gzip
from lxml import etree
from tqdm import tqdm
import argparse  


####################

parser = argparse.ArgumentParser(description='Process XML files.')

parser.add_argument('--work_dir', type=str, required=True, help='path to work directory')
parser.add_argument('--xml_files_dir', type=str, required=True, help='Path to the directory with XML files downloaded from NCBI')
parser.add_argument('--unique_pmids_pmcids', type=str, required=True, help='unique pmids and pmcids text file')
parser.add_argument('--output_file', type=str, required=True, help='name of output file')

args = parser.parse_args()

# Prepend work_dir to all the file paths
xml_files_dir = os.path.join(args.work_dir, args.xml_files_dir)
unique_pmids_pmcids = os.path.join(args.work_dir, args.unique_pmids_pmcids)
output_file = os.path.join(args.work_dir, args.output_file)

# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# xml_files_dir = os.path.join(work_dir, "ncbi_metadata_dir")
# unique_pmids_pmcids = os.path.join(work_dir, "unique_pmids_pmcids")
# output_file = os.path.join(work_dir, "ncbi_metadata_selection")

####################




# open list of pmids and pmcids (from text file)
# state how many unique we want the metadata about 


# create an empty df to accept content

# 


# Create an empty dataframe
df = pd.DataFrame(columns=["pmid_pmcids", "title", "abstract"])


























import os
import gzip
from lxml import etree
import pandas as pd
from tqdm import tqdm

path_to_xml_files = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir/test2"
id_filepath = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/unique_pmids_pmcids_test"  # This file should contain one PMID/PMCID per line.


# Read PMIDs and PMCIDs from file into a set for faster lookup
with open(id_filepath, 'r') as f:
    id_set = set(line.strip() for line in f)

# Filter XML files
files = [f for f in os.listdir(path_to_xml_files) if f.endswith(".xml.gz")]

# Lists to store extracted data
id_list = []
title_list = []
abstract_list = []

# Process files
for filename in tqdm(files, desc="Processing files"):
    
    filepath = os.path.join(path_to_xml_files, filename)
    with gzip.open(filepath, 'rt') as f:
        tree = etree.parse(f)
        root = tree.getroot()

        # Go through each article
        for article in tqdm(root.findall(".//PubmedArticle"), desc=f"Processing {filename}", leave=False):


            pmid_element = article.find(".//PMID")          
            pmcid_element = article.xpath(".//ArticleIdList/ArticleId[@IdType='PMC']")
            #pmcid_element = article.xpath(".//ArticleId[@IdType='PMC']")

            # Logic to handle both PMIDs and PMCIDs
            identifier = None
            if pmid_element is not None:
                identifier = pmid_element.text
                print(identifier)
            elif pmcid_element:
                identifier = pmcid_element[0].text
                print(identifier)

            # If the identifier is in the set, extract the information
            if identifier and identifier in id_set:
                
                title_element = article.find(".//ArticleTitle")
                title = title_element.text if title_element is not None else None

                abstract_element = article.find(".//AbstractText")
                abstract = abstract_element.text if abstract_element is not None else None

                # Append the data to the lists
                id_list.append(identifier)
                title_list.append(title)
                abstract_list.append(abstract)


# Convert lists to DataFrame
df = pd.DataFrame({
    "pmid_or_pmcid": id_list,
    "title": title_list,
    "abstract": abstract_list
})










import os
import gzip
from lxml import etree

path_to_xml_files = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir/test2"
files = [f for f in os.listdir(path_to_xml_files) if f.endswith(".xml.gz")]

results = []

# Iterate through XML files
for filename in files:
    filepath = os.path.join(path_to_xml_files, filename)
    with gzip.open(filepath, 'rt') as f:  # 'rt' mode is for reading text data.
        tree = etree.parse(f)
        
        # Iterate over each PubmedArticle
        for pubmed_article in tree.xpath("//PubmedArticle"):
            pmid = pubmed_article.xpath(".//MedlineCitation/PMID/text()")
            pmcid = pubmed_article.xpath(".//PubmedData/ArticleIdList/ArticleId[@IdType='pmc']/text()")
            
            # If both pmid and pmcid exist, append to results
            if pmid and pmcid:
                results.append({
                    "pmid": pmid[0],
                    "pmcid": pmcid[0]
                })

# Print the results
for item in results:
    print(f"PMID: {item['pmid']}, PMCID: {item['pmcid']}")














# =============================================================================
# 
# 
# # Filter files
# files = [f for f in os.listdir(path_to_xml_files) if f.endswith(".xml.gz")]
# print(f"Number of files: {len(files)}")
# 
# # Process: 
# no_files = 0
# no_pmid_in_files = 0
# no_found_pmid = 0
# 
# # Go through each file
# for filename in tqdm(files, desc="Processing files"):
#     
#     no_files+=1 
#     print("\nProcessing file number: ", no_files,'/',len(files))
#     
#     # Generate full file path
#     filepath = os.path.join(path_to_xml_files, filename)
#     with gzip.open(filepath, 'rt') as f:  # 'rt' mode is for reading text data.
#         tree = etree.parse(f)
#         root = tree.getroot()
# 
#         # Go through each article
#         for article in tqdm(root.findall(".//PubmedArticle"), desc=f"Processing {filename}", leave=False):
#             pmid_element = article.find(".//PMID")
#             if pmid_element is not None:
#                 pmid = int(pmid_element.text)  # Convert pmid to integer
# 
#                 no_pmid_in_files+=1
# 
#                 # If the PMID is in the list, extract the information
#                 if pmid in pmid_list:
#                     print("PMID ", pmid, " found in list:")  # Print if we found a match
# 
#                     no_found_pmid+=1
#                     title_element = article.find(".//ArticleTitle")
#                     if title_element is not None:
#                         title = title_element.text
#                     else:
#                         title = None
# 
#                     abstract_element = article.find(".//AbstractText")
#                     if abstract_element is not None:
#                         abstract = abstract_element.text
#                     else:
#                         abstract = None
# 
#                     # Add the data to the dataframe
#                     df = pd.concat([df, pd.DataFrame({"pmid_digits": [str(pmid)], "title": [title], "abstract": [abstract]})], ignore_index=True)
# 
# 
# print("total number of xml files processed over total: ", no_files, "/", len(files))
# print("total number of pmids from all processed xml files ", no_pmid_in_files)
# print("total number of pmids found in all files over total in our list", no_found_pmid, "/", len(pmid_list))
# 
# 
# 
# # merge data (biome, sample, pmid_digits) to this df (pmid, title, abstract):
# data['pmid_digits'] = data['pmid_digits'].astype(str)
# merged_df = pd.merge(data, df, on="pmid_digits", how="inner")  # inner gets only the common pmid_digits
# 
# 
# # Save the dataframe to a CSV file
# merged_df.to_csv(os.path.join(path_to_output_dir, 'sample_biome_pmid_title_abstract2.csv'), index=False)
# print("Output file succesfully written")
# 
# 
# =============================================================================




import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

def add_edges(graph, element, parent=None):
    node_name = element.tag
    graph.add_node(node_name)
    
    if parent is not None:
        graph.add_edge(parent, node_name)

    for child in element:
        add_edges(graph, child, node_name)

with open("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir/test2/pubmed23n1087.xml", "r") as f:
    tree = ET.parse(f)
    root = tree.getroot()
    
    graph = nx.DiGraph()
    add_edges(graph, root)

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, arrows=True, node_size=2000, node_color="skyblue", font_size=10)
    plt.show()









