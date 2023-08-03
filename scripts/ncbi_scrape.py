#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:32:32 2023

@author: dgaio
"""

# run as: 
# python github/metadata_mining/scripts/ncbi_scrape.py --output_dir "cloudstor/Gaio/MicrobeAtlasProject/ncbi_metadata_dir/"


import requests
from bs4 import BeautifulSoup
import urllib
from tqdm import tqdm
import os
import argparse  

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download XML files.')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the XML files.')
args = parser.parse_args()



url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"


home = os.path.expanduser('~')
ncbi_metadata_dir = os.path.join(home, args.output_dir)  

print('\nWriting output files here: ', ncbi_metadata_dir, '\n')

# Create the output directory if it doesn't exist
os.makedirs(ncbi_metadata_dir, exist_ok=True)  


response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
links = soup.select('a[href$=".xml.gz"]')  

total_files = len(links)
print(f"Total files to download: {total_files}")

for i, link in enumerate(links):
    file_url = urllib.parse.urljoin(url, link['href'])
    file_name = link['href']
    
    print('\n')
    print(f"Downloading file {i+1} of {total_files}: {file_name}")
    
    response = requests.get(file_url, stream=True)
    response.raise_for_status()  # Ensure we got a valid response
    
    # Progress bar
    file_size = int(response.headers.get('Content-Length', 0))
    progress = tqdm(response.iter_content(1024), f'Downloading {file_name}', total=file_size, unit='B', unit_scale=True, unit_divisor=1024)

    path_to_file = os.path.join(ncbi_metadata_dir, file_name) 
    with open(path_to_file, 'wb') as f:
        for data in progress.iterable:
            # Write data read to the file
            f.write(data)
            # Update the progress bar manually
            progress.update(len(data))

print("Files downloaded.")
# 1166 downloaded






















