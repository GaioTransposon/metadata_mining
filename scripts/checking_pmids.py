#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:46:49 2023

@author: dgaio
"""

import csv
from collections import defaultdict
import ast

input_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid.csv'
output_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/exploded_sample.info_biome_pmid.csv'

biome_counts = defaultdict(set)

with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=['sample', 'biome', 'pmid'])
    writer.writeheader()

    for row in reader:
        pmids_str = row['pmid']
        
        # Convert string representation of list to actual list
        try:
            pmids = ast.literal_eval(pmids_str)
        except (SyntaxError, ValueError):
            pmids = []

        if pmids:
            for pmid in pmids:
                writer.writerow({'sample': row['sample'], 'biome': row['biome'], 'pmid': pmid})
                biome_counts[row['biome']].add(pmid)
        else:
            writer.writerow(row)

# Print unique PMID counts
for biome, pmids in biome_counts.items():
    print(f"{biome}: {len(pmids)}")
    
    
    
    
    
    




import csv
from collections import defaultdict
import pandas as pd



# Define the path to your CSV file
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv'

# Use defaultdict to store counts
biome_counts = defaultdict(int)
seen_pmids = set()

# Open and read the CSV file
with open(file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        biome = row['biome']
        pmid = row['pmid']


        if pmid not in seen_pmids:
            biome_counts[biome] += 1
            seen_pmids.add(pmid)

# Print results
for biome, count in biome_counts.items():
    print(f"{biome}: {count}")
