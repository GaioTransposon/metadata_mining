#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:15:41 2023

@author: dgaio
"""



# cat sample.info_subset | grep -iE "bioproject|bio.project|PRJNA|PRJEB|PRJDB" | sort | uniq > test
# 40860 rows 


import re

# List of patterns to test
patterns = [
    r"PRJ[A-Z]+\s*\d+|\bbioproject[:/\s]\s*(\d+)\b"
]

# Read samples from a file
filename = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/test'
with open(filename, 'r') as f:
    samples = [line.strip() for line in f.readlines()]
n=0
for pattern in patterns:
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    
    for sample in samples:
        if not compiled_pattern.search(sample):
            n+=1
            print(f"Not captured by pattern: '{sample}'")

print(n, ' not found, out of ', len(samples))
# pretty good! I checked all the unmatched and they indeed don t make sense. 









