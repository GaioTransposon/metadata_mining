#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:08:44 2023

@author: dgaio
"""

# PART 1: 
# reads original ENVO.tsv
# cleans it, and outputs a dictionary of labels as keys and description as value. 

# PART 2: 
# goes through raw metadata files and replaces labels at each occurrence (regex flexible)
# with their respectective description. 


import pandas as pd
import re
import os


# Load the data from CSV
data = pd.read_csv('/Users/dgaio/Downloads/ENVO.tsv', sep='\t')

# Extract ENVO labels from the 'Term IRI' and 'Parent term IRI' columns
data['label'] = data['Term IRI'].str.split('/').str[-1]
data['Parent_label'] = data['Parent term IRI'].str.split('/').str[-1]

# Create a combined description column based on the presence of a definition
data['Joint_Info'] = data.apply(lambda row: f"{row['Term label']} (definition: {row['Definition']})" if pd.notna(row['Definition']) else row['Term label'], axis=1)

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


label_info_dict = result.set_index('label')['Joint_Info'].to_dict()

print(label_info_dict)


# Save the final dataframe to a new CSV file
output_df.to_csv('/Users/dgaio/Downloads/ENVO_parsed.tsv', index=False)













    def process_metadata(self, samples):
        shuffled_samples = samples.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        processed_samples_count = 0
        processed_samples_list = []  # To keep track of which samples have been processed
        
        # endings to filter out ("$" makes sure it will skip these if preceeded by a character or empty space)
        endings_to_remove = ["=$", "nan$", "not applicable$", "missing$", ". na$"]
        metadata_dict = {}
        
        for _, row in shuffled_samples.iterrows():
            metadata = self.fetch_metadata_from_sample(row['sample'])  # change this line as it's now a method
            
            print("Metadata for", row['sample'])
            
            processed_samples_count += 1
            processed_samples_list.append(row['sample'])
            print(f"Processed samples count: {processed_samples_count}")        
            
            cleaned_metadata_lines = []
            for line in metadata.splitlines():
                stripped_line = line.strip()  # strip whitespace
                should_keep = True
                if stripped_line.lower().startswith(("experiment", "run", ">")):
                    should_keep = False
                else:
                    for ending in endings_to_remove:
                        if re.search(ending, stripped_line, re.IGNORECASE):
                            print(f"Rejected line (ends with {ending}): {stripped_line}")
                            should_keep = False
                            break
                if should_keep:
                    cleaned_metadata_lines.append(stripped_line)
            cleaned_metadata = "\n".join(cleaned_metadata_lines)
            metadata_dict[row['sample']] = cleaned_metadata
            
            print("Cleaned metadata:")
            print(cleaned_metadata)
            print("===================================")
            
        print(f"All processed samples: {processed_samples_list}")
        
        return metadata_dict
    
    
    


def create_regex_pattern(label):
    # Extract the prefix and the digits from the label
    prefix, digits = re.match(r'([a-zA-Z]+)(\d+)', label).groups()
    
    # Construct a regex pattern
    pattern = prefix + r'\D+' + digits
    return pattern


base_dir = '/path/to/root/directory'  # Replace this with the path to the directory where your folders start

for dirpath, dirnames, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith('.txt'):  # Assuming the metadata files have .txt extension
            with open(os.path.join(dirpath, filename), 'r') as f:
                content = f.read()
                for label, joint_info in label_info_dict.items():
                    pattern = create_regex_pattern(label)
                    content = re.sub(pattern, joint_info, content, flags=re.IGNORECASE)
            with open(os.path.join(dirpath, filename.split('.')[0] + '_clean.txt'), 'w') as f:
                f.write(content)











