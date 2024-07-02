#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:17:15 2024

@author: dgaio
"""




# sub-biomes


import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from Levenshtein import distance as levenshtein_distance

# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'

# List of GPT files
gpt_json_files = [
    'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442_sbembeddings.json',
    'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500_sbembeddings.json'
]

# Gold dictionary file
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')

# Function to load embeddings and extract sub-biome assignments
def extract_sub_biomes(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    sub_biomes = {k: v['sub-biome'] for k, v in data.items()}
    return sub_biomes

# Extract sub-biome assignments for GPT files
gpt_sub_biomes = {}
for gpt_file in gpt_json_files:
    gpt_json_path = os.path.join(embeddings_dir, gpt_file)
    gpt_sub_biomes[gpt_file] = extract_sub_biomes(gpt_json_path)

# Extract sub-biome assignments for gold dictionary
gold_sub_biomes = extract_sub_biomes(gold_dict_json_path)

# Extract sample IDs common to all
common_sample_ids = set(gold_sub_biomes.keys())
for gpt_sub_biome in gpt_sub_biomes.values():
    common_sample_ids &= set(gpt_sub_biome.keys())

# Prepare data for TF-IDF vectorization
def prepare_data(gold_sub_biomes, gpt_sub_biomes, common_sample_ids):
    gold_texts = []
    gpt_texts = []
    for sample_id in common_sample_ids:
        gold_texts.append(gold_sub_biomes[sample_id])
        gpt_texts.append(gpt_sub_biomes[sample_id])
    return gold_texts, gpt_texts

# Function to calculate TF-IDF cosine similarity
def calculate_cosine_similarity(gold_texts, gpt_texts):
    vectorizer = TfidfVectorizer().fit(gold_texts + gpt_texts)
    gold_vectors = vectorizer.transform(gold_texts)
    gpt_vectors = vectorizer.transform(gpt_texts)
    cosine_similarities = cosine_similarity(gold_vectors, gpt_vectors)
    return cosine_similarities.diagonal().mean()

# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(gold_texts, gpt_texts):
    def tokenize(text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
        return set(tokens)

    gold_sets = [tokenize(text) for text in gold_texts]
    gpt_sets = [tokenize(text) for text in gpt_texts]
    
    mlb = MultiLabelBinarizer()
    gold_binarized = mlb.fit_transform(gold_sets)
    gpt_binarized = mlb.transform(gpt_sets)
    
    jaccard_similarities = []
    for gold, gpt in zip(gold_binarized, gpt_binarized):
        jaccard_sim = jaccard_score(gold, gpt, average='macro')
        jaccard_similarities.append(jaccard_sim)
    
    return np.mean(jaccard_similarities)

# Function to calculate Levenshtein similarity
def calculate_levenshtein_similarity(gold_texts, gpt_texts):
    levenshtein_similarities = []
    for gold, gpt in zip(gold_texts, gpt_texts):
        max_len = max(len(gold), len(gpt))
        if max_len == 0:
            levenshtein_similarities.append(1.0)
        else:
            levenshtein_sim = 1 - levenshtein_distance(gold, gpt) / max_len
            levenshtein_similarities.append(levenshtein_sim)
    return np.mean(levenshtein_similarities)

# Calculate similarity scores for each GPT file
scores = {}
for gpt_file in gpt_json_files:
    gold_texts, gpt_texts = prepare_data(gold_sub_biomes, gpt_sub_biomes[gpt_file], common_sample_ids)
    
    cosine_sim = calculate_cosine_similarity(gold_texts, gpt_texts)
    jaccard_sim = calculate_jaccard_similarity(gold_texts, gpt_texts)
    levenshtein_sim = calculate_levenshtein_similarity(gold_texts, gpt_texts)
    
    scores[gpt_file] = {
        'cosine_similarity': cosine_sim,
        'jaccard_similarity': jaccard_sim,
        'levenshtein_similarity': levenshtein_sim,
        'average_similarity': np.mean([cosine_sim, jaccard_sim, levenshtein_sim])
    }

# Print results
for gpt_file, score in scores.items():
    print(f"Scores for {gpt_file}:")
    print(f"  Cosine Similarity: {score['cosine_similarity']:.2f}")
    print(f"  Jaccard Similarity: {score['jaccard_similarity']:.2f}")
    print(f"  Levenshtein Similarity: {score['levenshtein_similarity']:.2f}")
    print(f"  Average Similarity: {score['average_similarity']:.2f}")









# what about biomes

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from Levenshtein import distance as levenshtein_distance
import csv


# Directory containing the JSON files
embeddings_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/embeddings'
parent_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'  # Assuming files are one directory up from embeddings

# List of GPT files
gpt_files = [
    'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.0_presp1.5_rs22_API120_normal_dt202406051442.txt',
    'gpt_clean_output_nspb100_chunkingyes_chunksize2000_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp2.0_presp1.5_rs22_API153_normal_dt202406051500.txt'
]

# Gold dictionary file
gold_dict_json_path = os.path.join(embeddings_dir, 'gold_dict_sbembeddings.json')

# Function to load embeddings and extract sub-biome assignments from JSON
def extract_sub_biomes_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    sub_biomes = {k: v['biome'] for k, v in data.items()}
    return sub_biomes

# Function to extract sub-biomes from .txt file (4th column)
def extract_sub_biomes_from_txt(txt_file_path):
    sub_biomes = {}
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            sample_id = row[0].strip()
            sub_biome = row[1].strip()  # Assuming 4th column is the sub-biome
            sub_biomes[sample_id] = sub_biome
    return sub_biomes

# Extract sub-biome assignments for GPT files
gpt_sub_biomes = {}
for gpt_file in gpt_files:
    txt_file_path = os.path.join(parent_dir, gpt_file)
    gpt_sub_biomes[gpt_file] = extract_sub_biomes_from_txt(txt_file_path)

# Extract sub-biome assignments for gold dictionary
gold_sub_biomes = extract_sub_biomes_from_json(gold_dict_json_path)

# Extract sample IDs common to all
common_sample_ids = set(gold_sub_biomes.keys())
for gpt_sub_biome in gpt_sub_biomes.values():
    common_sample_ids &= set(gpt_sub_biome.keys())

# Prepare data for TF-IDF vectorization
def prepare_data(gold_sub_biomes, gpt_sub_biomes, common_sample_ids):
    gold_texts = []
    gpt_texts = []
    for sample_id in common_sample_ids:
        gold_texts.append(gold_sub_biomes[sample_id])
        gpt_texts.append(gpt_sub_biomes[sample_id])
    return gold_texts, gpt_texts

# Function to calculate TF-IDF cosine similarity
def calculate_cosine_similarity(gold_texts, gpt_texts):
    vectorizer = TfidfVectorizer().fit(gold_texts + gpt_texts)
    gold_vectors = vectorizer.transform(gold_texts)
    gpt_vectors = vectorizer.transform(gpt_texts)
    cosine_similarities = cosine_similarity(gold_vectors, gpt_vectors)
    return cosine_similarities.diagonal().mean()

# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(gold_texts, gpt_texts):
    def tokenize(text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
        return set(tokens)

    gold_sets = [tokenize(text) for text in gold_texts]
    gpt_sets = [tokenize(text) for text in gpt_texts]
    
    mlb = MultiLabelBinarizer()
    gold_binarized = mlb.fit_transform(gold_sets)
    gpt_binarized = mlb.transform(gpt_sets)
    
    jaccard_similarities = []
    for gold, gpt in zip(gold_binarized, gpt_binarized):
        jaccard_sim = jaccard_score(gold, gpt, average='macro')
        jaccard_similarities.append(jaccard_sim)
    
    return np.mean(jaccard_similarities)

# Function to calculate Levenshtein similarity
def calculate_levenshtein_similarity(gold_texts, gpt_texts):
    levenshtein_similarities = []
    for gold, gpt in zip(gold_texts, gpt_texts):
        max_len = max(len(gold), len(gpt))
        if max_len == 0:
            levenshtein_similarities.append(1.0)
        else:
            levenshtein_sim = 1 - levenshtein_distance(gold, gpt) / max_len
            levenshtein_similarities.append(levenshtein_sim)
    return np.mean(levenshtein_similarities)

# Calculate similarity scores for each GPT file
scores = {}
for gpt_file in gpt_files:
    gold_texts, gpt_texts = prepare_data(gold_sub_biomes, gpt_sub_biomes[gpt_file], common_sample_ids)
    
    cosine_sim = calculate_cosine_similarity(gold_texts, gpt_texts)
    jaccard_sim = calculate_jaccard_similarity(gold_texts, gpt_texts)
    levenshtein_sim = calculate_levenshtein_similarity(gold_texts, gpt_texts)
    
    scores[gpt_file] = {
        'cosine_similarity': cosine_sim,
        'jaccard_similarity': jaccard_sim,
        'levenshtein_similarity': levenshtein_sim,
        'average_similarity': np.mean([cosine_sim, jaccard_sim, levenshtein_sim])
    }

# Print results
for gpt_file, score in scores.items():
    print(f"Scores for {gpt_file}:")
    print(f"  Cosine Similarity: {score['cosine_similarity']:.2f}")
    print(f"  Jaccard Similarity: {score['jaccard_similarity']:.2f}")
    print(f"  Levenshtein Similarity: {score['levenshtein_similarity']:.2f}")
    print(f"  Average Similarity: {score['average_similarity']:.2f}")


