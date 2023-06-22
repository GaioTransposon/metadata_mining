#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:59:10 2023

@author: dgaio
"""

import pandas as pd

# Read the file
df = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv', sep='\t')

# Split the column
df[['sample_id_run', 'sample_id']] = df['SampleID'].str.split('.', expand=True)

# Drop the original column
df = df.drop(columns='SampleID')

# Save the DataFrame
df.to_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged_sep_cols.tsv', sep='\t', index=False)



import pandas as pd

# Read the IDs from the first file
df = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged_sep_cols.tsv', sep='\t')
sample_ids = df['sample_id'].tolist()

# Prepare to read the large file line by line
with open("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info", "r") as large_file, open("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_otu_97_cleanedEnvs_bray_maxBray08_nproj10_20210224_merged.tsv", "w") as subset_file:
    write_lines = False
    for line in large_file:
        if line.startswith('>'):  # Indicates start of a new sample
            sample_id = line[1:].strip()  # Remove the leading '>'
            if sample_id in sample_ids:
                write_lines = True
                subset_file.write(line)
            else:
                write_lines = False
        elif write_lines:  # If it's part of a sample we're interested in
            subset_file.write(line)
