#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:34:22 2023

@author: dgaio
"""

import pandas as pd

def extract_sample_related_fields(file_path):
    with open(file_path, 'r') as file:
        data_dict = {}
        key = None
        terms = ["sample_common_name", "sample_dev_stage", "sample_host scientific name", 
                 "sample_isolate", "sample_isolation_source", "sample_env_biome", 
                 "sample_env_broad_scale", "sample_env_feature", "sample_env_local_scale", 
                 "sample_env_material", "sample_env_medium", "sample_host", 
                 "sample_tissue", "sample_scientific_name", "species"]

        for line in file:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
                data_dict[key] = ''
            elif line == '':
                continue
            else:
                term, _, value = line.partition("=")
                term = term.strip().lower()
                if term in terms:
                    if data_dict[key] == '':
                        data_dict[key] = value.strip()
                    else:
                        data_dict[key] += '; ' + value.strip()

    df = pd.DataFrame(list(data_dict.items()), columns=['Entry', 'Data'])
    df['Data'] = df['Data'].str.replace('_', ' ')
    return df



file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_400000'
df = extract_sample_related_fields(file_path)
print(df)








# =============================================================================
# # parsing of original ENVO.tsv 
# # File path
# file_path = "/Users/dgaio/Downloads/ENVO.tsv"
# 
# 
# # Read the TSV file into a DataFrame
# df = pd.read_csv(file_path, sep='\t')
# 
# # Filter rows with "ENVO" labels
# df = df[df['Term IRI'].str.contains('ENVO')]
# 
# # Extract the term and parent term labels
# df['term'] = df['Term IRI'].str.split('/').str[-1].str.replace('ENVO_', 'ENVO:')
# df['term_label'] = df['Term label']
# df['parent_term'] = df['Parent term IRI'].str.split('/').str[-1].str.replace('ENVO_', 'ENVO:')
# df['parent_label'] = df['Parent term label']
# 
# # Drop the unnecessary columns
# df = df[['term', 'term_label', 'parent_term', 'parent_label']]
# 
# # Print the resulting DataFrame
# print(df)
# df.to_csv('/Users/dgaio/github/metadata_mining/middle_dir/envo_parsed.tsv', index=False)
# 
# =============================================================================





import pandas as pd


envo_df = pd.read_csv("/Users/dgaio/github/metadata_mining/middle_dir/envo_parsed.tsv")



def replace_terms(df, column, envo_df):
    # Create dictionaries for mapping
    term_dict = pd.Series(envo_df.term_label.values, index=envo_df.term).to_dict()
    parent_term_dict = pd.Series(envo_df.parent_label.values, index=envo_df.parent_term).to_dict()
    
    # Combine the two dictionaries
    combined_dict = {**term_dict, **parent_term_dict}
    print(combined_dict)
    # Replace terms in the specified column
    df[column] = df[column].replace(combined_dict)
    return df










df2 = replace_terms(df, 'Data', envo_df)







import string
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def most_common_words(df, column, n):
    all_words = ' '.join(df[column]).lower()
    # Remove punctuation
    all_words = all_words.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = word_tokenize(all_words)
    freq_dist = FreqDist(tokenized_words)
    return freq_dist.most_common(n)




# add envo to these. 
# first parse ENVO.tst to get: envo label - term. 

bacteria = ["salmonella", "enterica", "streptococcus", "plasmodium", "falciparum", 
            "staphylococcus", "aureus", "campylobacter", "escherichia", "coli"]


soil = ["soil", "mud", "dirt", "clay", "agricultural", "field", "root"]  # Add your specific words

water = ["water", "freshwater", "aquatic", "lake", "river", "ocean", "marine", "sediment", "wastewater", 
         "mediterraneal", "sea", "seawater", "pond", "sludge", "groundwater"]


plant = ["forest", "plant", "leaf", "tree", "flower", "grass", "rhizosphere", "roots", "hordeum", "quercus", "cotton"]


animal = ["homo", "sapiens", "bodily", "skin", "lung", "oral", "blood", "vaginal", 
          "gastrointestinal", "tract", "intestinal", "feces", "feces", "excreta", "stool", "gut", "envo00002003", 
          "respiratory", "tissue",
          "mus", "musculus", "mouse" , "vulpes", "sus", "scrofa", "pig", "squirrel", 
          "mytilus", "galloprovincialis", "lucinid",  "invertebrate", "coronavirus", "mollusc", 
          "cat", "dog", "bird", "dicentrarchus", "labrax", "malaria", "danio", "rerio", "aphis"]







def assign_biome(row):
    for word in word_tokenize(row.lower()):
        if word in soil:
            return "Soil"
        if word in water:
            return "Water"
        if word in plant:
            return "Plant"
        if word in animal:
            return "Animal"
        if word in bacteria:
            return "Bacteria"
    return "Unknown"  # return 'Unknown' if no match is found


















df['biome'] = df['Data'].apply(assign_biome)



unknown_rows = df.loc[df['biome'] == 'Unknown']


most_common_n = most_common_words(unknown_rows, 'Data', 100)  # Adjust n as needed

for word, count in most_common_n:
    print(word, count)

