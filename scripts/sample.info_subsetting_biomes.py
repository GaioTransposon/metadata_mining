#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:59:10 2023

@author: dgaio
"""


import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

df = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_pmid_biome.csv')

# remove NaN rows
df = df.dropna(subset=['pmid_digits'])

# remove unknown biome
df = df[df['EnvClean_merged'] != 'unknown']

# subset to minim number of non-NaN and not-unknown per biome: 283 per biome. 
df = df.groupby('EnvClean_merged').apply(lambda x: x.sample(min(len(x), 283))).reset_index(drop=True)


# extract title and abstract
# Create new columns for the title and abstract
df['title'] = ''
df['abstract'] = ''

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    # Construct the URL for the PubMed article
    url = "https://pubmed.ncbi.nlm.nih.gov/" + str(row['pmid_digits'])
    
    # Send a GET request and pause for 5 seconds between each request
    response = requests.get(url)
    time.sleep(5)

    # Only parse the page if the status code is 200
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the title and abstract
        title = soup.find('h1', attrs={'class' : 'heading-title'})
        abstract = soup.find('div', attrs={'class' : 'abstract-content selected'})
        
        # Some pages might not have a title or an abstract
        df.at[i, 'title'] = title.text.strip() if title else "Not found"
        df.at[i, 'abstract'] = abstract.text.strip() if abstract else "Not found"

print(df)


