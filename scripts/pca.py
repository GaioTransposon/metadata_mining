#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:55:34 2023

@author: dgaio
"""

# PCA 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler

def extract_nouns(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    nouns = [word for word, pos in tagged_tokens if pos.startswith('NN')]
    return ' '.join(nouns)

def preprocess_text(text):
    # Remove punctuation and convert to lowercase if necessary
    # Add any additional preprocessing steps here if needed
    return text


# Example usage
# Assuming you have a dataframe named 'data' with columns 'confirmed_biome', 'title', and 'abstract'
file_path = os.path.expanduser('~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv')

# Step 1: Collect and prepare the data
data = pd.read_csv(file_path).dropna(subset=['confirmed_biome'])





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_features(data):
    # Combine 'title' and 'abstract' into a single text column
    data['text'] = data['title'] + ' ' + data['abstract']
    
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit and transform the text data
    word_count_matrix = vectorizer.fit_transform(data['text'])
    
    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the labels
    labels = data['confirmed_biome']
    
    return word_count_matrix, feature_names, labels


def visualize_pca(pca_matrix, labels):
    # Create a scatter plot with colored labels
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'purple']  # Define colors for the categories

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        plt.scatter(pca_matrix[indices, 0], pca_matrix[indices, 1], color=colors[i], label=label)

    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()


# Example usage
file_path = os.path.expanduser('~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv')

# Step 1: Collect and prepare the data
data = pd.read_csv(file_path).dropna(subset=['confirmed_biome'])

# Step 2: Extract features
word_count_matrix, feature_names, labels = extract_features(data)

# Step 3: Perform PCA
pca = PCA(n_components=10)
pca_matrix = pca.fit_transform(word_count_matrix.toarray())

# Step 4: Visualize PCA
visualize_pca(pca_matrix, labels)

