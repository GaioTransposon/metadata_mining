#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:34:22 2023

@author: dgaio
"""

import pandas as pd
import re



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



# Standardize the labels in column 'Data'
df['Data'] = df['Data'].str.replace(r'ENVO[ \-]+', 'ENVO:', regex=True)


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





envo_df = pd.read_csv("/Users/dgaio/github/metadata_mining/middle_dir/envo_parsed.tsv")


def replace_terms_with_changes(df, envo_df):
    # Create a dictionary from the envo_df DataFrame
    term_dict = pd.Series(envo_df.parent_label.values, index=envo_df.term).to_dict()

    # A function to replace ENVO terms with corresponding parent label
    def replace_with_dict(row, term_dict):
        text = row['Data']
        changes = []
        for term in re.findall(r'\bENVO:\d+\b', text):
            if term in term_dict:
                changes.append((term, term_dict[term]))
                text = text.replace(term, term_dict[term])
        return {'Data': text, 'Changes': changes}

    # Replace the terms in the Data column and keep track of changes
    changes_df = df.apply(replace_with_dict, term_dict=term_dict, axis=1).apply(pd.Series)
    
    return changes_df






df2 = replace_terms_with_changes(df, envo_df)

print(df2['Changes'])
(df2['Changes'].str.len() == 0).sum()
(df2['Changes'].str.len() > 0).sum()


# which envo are left? 
# Suppose you have a DataFrame called 'df' and want to return rows with partial matches in column 'column_name'
partial_match = df2['Data'].str.contains('ENVO', case=False)


check = df2[partial_match]
print("if right, it should be empty: ", len(check))












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


mobilis 48
shigella 42
listeria 41
monocytogenes 40
mycobacterium 38
tuberculosis 35
sonnei
culture 27
bacterial
bacterium
neisseria 28




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


















df2['biome'] = df2['Data'].apply(assign_biome)



unknown_rows = df2.loc[df2['biome'] == 'Unknown']


most_common_n = most_common_words(unknown_rows, 'Data', 100)  # Adjust n as needed

for word, count in most_common_n:
    print(word, count)

