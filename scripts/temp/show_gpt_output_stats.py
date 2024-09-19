#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:49:20 2023

@author: dgaio
"""




import os
import openai
import pandas as pd
import numpy as np  
from collections import Counter
import argparse  
import pickle
import re
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# for testing purposes
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
input_df = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")


########### # 2. open gold_dict and transform to a df
with open(input_gold_dict, 'rb') as file:
    input_gold_dict = pickle.load(file)

input_gold_dict = input_gold_dict[0] # this is because the second item is the list of pmids I processed - it was necessary to run confirm_biome_game.py, now not necessary anymore 
gold_dict_df = pd.DataFrame(input_gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)
###########


def process_and_merge(input_file, response_df, gold_dict_df):
    """
    Read and process the input file, then merge with the provided dataframes.

    Args:
    - input_file (str): Path to the input file to read.
    - response_df (pd.DataFrame): The dataframe containing the cleaned responses.
    - gold_dict_df (pd.DataFrame): The dataframe containing the gold dictionary data.

    Returns:
    - pd.DataFrame: The merged dataframe.
    """
    # Read the input dataframe
    input_df = pd.read_csv(input_file)
    input_df['biome'] = input_df['biome'].replace('aquatic', 'water')
    input_df = input_df[['sample', 'biome']]
    
    # Merge with the gold dictionary
    gold_dict_input_df = pd.merge(gold_dict_df, input_df, on='sample', how='inner')
    
    # Merge with the cleaned responses
    merged_df = pd.merge(response_df, gold_dict_input_df, on='sample', how='inner')
    
    return merged_df



def plot_confusion_matrix(df, actual_col='curated_biome', predicted_col='gpt_generated_biome'):
    """
    Plot a confusion matrix heatmap based on actual and predicted values.

    Args:
    - df (pd.DataFrame): DataFrame with actual and predicted values.
    - actual_col (str): Name of the column in df that contains the actual values.
    - predicted_col (str): Name of the column in df that contains the predicted values.

    Returns:
    - None
    """
    # Identify unique labels from both columns
    labels = sorted(list(set(df[actual_col]).union(set(df[predicted_col]))))

    # Generate the confusion matrix with explicit label ordering
    matrix = confusion_matrix(df[actual_col], df[predicted_col], labels=labels)

    # Normalize the matrix to get percentages
    normalized_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] * 100

    # Create combined annotations with percentages and raw counts
    annotations = [["{0:.2f}%\n(n={1})".format(normalized_matrix[i, j], matrix[i, j]) 
                    for j in range(len(matrix[i]))] for i in range(len(matrix))]

    # Plot the heatmap
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=labels, 
                yticklabels=labels)
    plt.xlabel('Actual biome')
    plt.ylabel('Predicted biome')
    plt.title('Confusion Matrix for Predictions vs Actual Biomes')
    plt.show()






# Read the file into a dataframe
file_name = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gtp_35_output_20231010_160709.txt"
dfr = pd.read_csv(file_name, sep=": ", engine='python', header=None, names=["sample", "gpt_generated_biome"])
# Display the dataframe
print(dfr)
len(dfr)


# Usage:
result_df = process_and_merge(input_df, dfr, gold_dict_df)
print(result_df)

# Usage:
plot_confusion_matrix(result_df)






































