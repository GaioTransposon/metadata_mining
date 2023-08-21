#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:00:41 2023

@author: dgaio
"""


# conda install xmltodict
# conda install -c conda-forge pygraphviz
# conda install -c conda-forge networkx


# # run as: 
# python ~/github/metadata_mining/scripts/get_ncbi_metadata.py  \
#         --work_dir '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/' \
#             --sample_info_biome_pmid "sample.info_biome_pmid.csv" \
#                 --xml_files_dir "ncbi_metadata_dir" \
#                         --output_file "sample.info_biome_pmid_title_abstract.csv" \
#                             --pmids_not_found_file "pmids_not_found_file.pkl" \
#                                 --figure "sample.info_biome_pmid_title_abstract.pdf"
    
 

import pandas as pd
import os
import gzip
import argparse  
import ast
import xml.etree.ElementTree as ET
import time
import pickle
import matplotlib.pyplot as plt

####################

parser = argparse.ArgumentParser(description='Process XML files.')

parser.add_argument('--work_dir', type=str, required=True, help='path to work directory')
parser.add_argument('--sample_info_biome_pmid', type=str, required=True, help='path to input df')
parser.add_argument('--xml_files_dir', type=str, required=True, help='Path to the directory with XML files downloaded from NCBI')
parser.add_argument('--output_file', type=str, required=True, help='name of output file')
parser.add_argument('--pmids_not_found_file', type=str, required=True, help='name of file where pmids for which (either) no title, abstract, or neither were found')
parser.add_argument('--figure', type=str, required=True, help='name of figure file')

args = parser.parse_args()

# Prepend work_dir to all the file paths
xml_files_dir = os.path.join(args.work_dir, args.xml_files_dir)
sample_info_biome_pmid = os.path.join(args.work_dir, args.sample_info_biome_pmid)
output_file = os.path.join(args.work_dir, args.output_file)
pmids_not_found_file = os.path.join(args.work_dir, args.pmids_not_found_file)
figure = os.path.join(args.work_dir, args.figure)


# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# sample_info_biome_pmid = os.path.join(work_dir, "sample.info_biome_pmid.csv")
# xml_files_dir = os.path.join(work_dir, "ncbi_metadata_dir/test2")
# output_file = os.path.join(work_dir, "ncbi_metadata_selection")
# pmids_not_found_file = os.path.join(work_dir, "pmids_not_found_file")
# figure = os.path.join(work_dir, "figure")
####################



############ 1. open input df
s = pd.read_csv(sample_info_biome_pmid)

############ 2. get a list of unique PMIDs

# Convert string representation of lists to actual lists
s['pmid'] = s['pmid'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

# Flatten the lists and get unique PMIDs
unique_pmids = pd.Series([pmid for sublist in s['pmid'].dropna() for pmid in sublist]).unique().tolist()
len(unique_pmids)



############ 3. go through files and fill in using list of unique PMIDs

def extract_text(element):
    """Extracts and concatenates text from an XML element and all its descendants."""
    if element is None:
        return None
    
    # Start with the text of the current element, or an empty string if it's None
    parts = [element.text or '']
    
    # Iterate over child elements and append their text
    for child in element:
        parts.append(extract_text(child))
        if child.tail:
            parts.append(child.tail)
    
    # Join all parts and return
    return ''.join(parts)





# Sample list of unique PMIDs
unique_pmids_set = set(unique_pmids)

# Create a list to accumulate data
data = []

pmids_found = set()
pmid_found_no_title = set()
pmid_found_no_abstract = set()
pmid_found_no_title_no_abstract = set()

file_counter = 0
start_time = time.time()
total_files = sum(1 for filename in os.listdir(xml_files_dir) if filename.endswith('.xml.gz'))
print('Total xml files: ', total_files)

for filename in os.listdir(xml_files_dir):
    if filename.endswith(".xml.gz"):
        with gzip.open(os.path.join(xml_files_dir, filename), 'rt') as file:
            tree = ET.parse(file)
            root = tree.getroot()

            for article in root.findall(".//PubmedArticle"):
                pmid_element = article.find(".//PMID")
                if pmid_element is not None and pmid_element.text in unique_pmids_set:
                    title_element = article.find(".//ArticleTitle")
                    abstract_element = article.find(".//AbstractText")
                    
                    
                    title = extract_text(title_element)
                    abstract = extract_text(abstract_element)
                    #title = title_element.text if title_element is not None else None
                    #abstract = abstract_element.text if abstract_element is not None else None

                    if title and abstract:
                        data.append({
                            "pmid": pmid_element.text,
                            "title": title,
                            "abstract": abstract
                        })
                        pmids_found.add(pmid_element.text)
                    else:
                        if not title and not abstract:
                            pmid_found_no_title_no_abstract.add(pmid_element.text)
                        elif not title:
                            pmid_found_no_title.add(pmid_element.text)
                        elif not abstract:
                            pmid_found_no_abstract.add(pmid_element.text)
                            
        file_counter += 1
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / file_counter
        estimated_time_left = avg_time_per_file * (total_files - file_counter)
        
        print(f"Processed {file_counter}/{total_files} XML files. Estimated time left: {estimated_time_left:.2f} seconds")


# Convert accumulated data to a DataFrame
df_accommodate_metadata = pd.DataFrame(data)

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")

pmids_not_found = unique_pmids_set - pmids_found - pmid_found_no_title - pmid_found_no_abstract - pmid_found_no_title_no_abstract

pmids_not_found = list(pmids_not_found)
pmid_found_no_title = list(pmid_found_no_title)
pmid_found_no_abstract = list(pmid_found_no_abstract)
pmid_found_no_title_no_abstract = list(pmid_found_no_title_no_abstract)



# Bundle lists into a dictionary for easier deserialization
data = {
    'pmids_not_found': pmids_not_found,
    'pmid_found_no_title': pmid_found_no_title,
    'pmid_found_no_abstract': pmid_found_no_abstract,
    'pmid_found_no_title_no_abstract': pmid_found_no_title_no_abstract
}
# Save data to a pickle file
with open(pmids_not_found_file, 'wb') as f:
    pickle.dump(data, f)


############ 4. Merge ncbi metadata to our original dataframe:  


# Step 1: Expand the 's' dataframe
s_exploded = s.explode('pmid')

# Step 2: Merge with 'df_accommodate_metadata'
merged_df = s_exploded.merge(df_accommodate_metadata, on='pmid', how='left')

# Step 3: Post-process (if needed)
merged_df = merged_df[['sample', 'biome', 'pmid', 'title', 'abstract']]  # Reorder columns

# Save
merged_df.to_csv(os.path.join(output_file), index=False)
print("Output file succesfully written")



############ 5. Draw plot (reports stats)


# Create indicator columns
merged_df['has_info'] = merged_df.apply(lambda x: 1 if pd.notna(x['title']) and pd.notna(x['abstract']) else 0, axis=1)
merged_df['missing_info'] = merged_df.apply(lambda x: 1 if pd.isna(x['title']) or pd.isna(x['abstract']) else 0, axis=1)

# Group by biome and compute sum of the indicators and unique PMIDs count for samples with both title and abstract
biome_summary = merged_df.groupby('biome').agg({
    'has_info': 'sum',
    'missing_info': 'sum',
    'pmid': lambda x: x[x.notna()].nunique()  # Count unique PMIDs with non-NaN values
}).reset_index()
biome_summary.columns = ['biome', 'has_info', 'missing_info', 'unique_pmids']

# Plotting
ax = biome_summary.set_index('biome').plot(kind='bar', figsize=(10, 6), position=0.5)

plt.title("Sample Counts by Biome and Content Availability")
plt.ylabel("Number of Samples/Unique PMIDs")
plt.xlabel("Biome")
plt.xticks(rotation=0)

# Annotating bars with the counts on top
for p in ax.patches:
    ax.annotate(str(p.get_height()), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points', 
                rotation=90)

plt.tight_layout()

# Save the plot
plt.savefig(figure, dpi=300, bbox_inches='tight')

plt.show()



############




