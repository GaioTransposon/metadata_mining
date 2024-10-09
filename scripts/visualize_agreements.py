#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:02:23 2024

@author: dgaio
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Results:

def plot_data(df, plot_title):
    
    
    df = df.copy()
    
    columns_to_keep = [
        'Label', 
        'Agreement biome (exact match)', 
        'Agreement biome (lenient match)', 
        'Average Similarity', 
        'Median Similarity', 
        'Standard Deviation', 
    ]
    df = df[columns_to_keep]

    # cols new names
    column_rename_map = {
        'Agreement biome (exact match)': 'agreement_exact',
        'Agreement biome (lenient match)': 'agreement_lenient',
        'Average Similarity': 'avg_sim',
        'Median Similarity': 'med_sim',
        'Standard Deviation': 'SD',
    }
    df.rename(columns=column_rename_map, inplace=True)

    # extract percentages
    df['agreement_exact'] = df['agreement_exact'].str.extract(r'(\d+\.\d+)%').astype(float)
    df['agreement_lenient'] = df['agreement_lenient'].str.extract(r'(\d+\.\d+)%').astype(float)
    
    df.set_index('Label', inplace=True)

    # convert to percentages
    df['avg_sim'] = df['avg_sim'] * 100
    df['med_sim'] = df['med_sim'] * 100
    df['SD'] = df['SD'] * 100  

    # plot
    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.8  # space between bars 
    n = len(df.columns) 
    indices = np.arange(len(df))  
    
    for i, column in enumerate(df.columns):
        ax.bar(indices - width/2. + i/float(n)*width, df[column], width=width/float(n), label=column)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(['' if str(label) == 'nan' else label for label in df.index], rotation=90)
    
    plt.xlabel('')
    plt.ylabel('values')
    plt.title(plot_title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=8) 
    
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()




file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_results.csv'
df_full = pd.read_csv(file_path)


df_subset = df_full[0:7] 
plot_data(df_subset, 'chunking y/n + chunk sizes')  

df_subset = df_full[8:14] 
plot_data(df_subset, 'models')  


df_subset = df_full[15:74] 
plot_data(df_subset, 'creativity params')  


df_subset = df_full[75:] 
plot_data(df_subset, 'sync vs async + reproducibility + robustness')  




################################################################################
################################################################################
################################################################################



# Stats:

def process_and_visualize(df, my_title, cell_font_size, labels_font):
    df = df.dropna()

    # Define the columns to keep
    columns_to_keep = ['Label1', 'Label2', 'P-value', 'Adjusted P-value', 'Test Type', 'validation']
    df = df[columns_to_keep]

    # Create a unique set of labels (keeping original order)
    labels = pd.unique(df[['Label1', 'Label2']].to_numpy().flatten())

    # Initialize matrices for P-values and annotations
    biome_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    subbiome_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    biome_annot = pd.DataFrame("", index=labels, columns=labels)
    subbiome_annot = pd.DataFrame("", index=labels, columns=labels)

    # Populate the matrices
    for _, row in df.iterrows():
        label1, label2 = row['Label1'], row['Label2']
        p_value = row['P-value']
        adj_p_value = row['Adjusted P-value']
        validation_type = row['validation']
        annotation = f"{p_value:.2f};\n{adj_p_value:.2f}"


        if validation_type == 'biome':
            biome_matrix.at[label1, label2] = adj_p_value
            biome_matrix.at[label2, label1] = adj_p_value  # mirroring for biome
            biome_annot.at[label1, label2] = annotation
            biome_annot.at[label2, label1] = annotation

        elif validation_type == 'sub-biome':
            subbiome_matrix.at[label1, label2] = adj_p_value
            subbiome_matrix.at[label2, label1] = adj_p_value  # mirroring for sub-biome
            subbiome_annot.at[label1, label2] = annotation
            subbiome_annot.at[label2, label1] = annotation
         
                        
    # Merge matrices: biome in the lower triangle, sub-biome in the upper triangle
    combined_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    combined_annotations = pd.DataFrame("", index=labels, columns=labels)
    for label1 in labels:
        for label2 in labels:
            if labels.tolist().index(label1) < labels.tolist().index(label2):
                combined_matrix.at[label1, label2] = subbiome_matrix.at[label1, label2]
                combined_annotations.at[label1, label2] = subbiome_annot.at[label1, label2]
            elif labels.tolist().index(label1) > labels.tolist().index(label2):
                combined_matrix.at[label1, label2] = biome_matrix.at[label1, label2]
                combined_annotations.at[label1, label2] = biome_annot.at[label1, label2]
            else:
                combined_matrix.at[label1, label2] = '-'  # for diagonal
                combined_annotations.at[label1, label2] = '-'


    # Plotting
    numeric_combined_matrix = combined_matrix.replace('-', np.nan).astype(float)
    mask_upper = np.triu(np.ones_like(numeric_combined_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(numeric_combined_matrix, dtype=bool), k=-1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_combined_matrix, mask=mask_upper, cmap='Greens', annot=combined_annotations, fmt="s", cbar=False,
                linewidths=.2, linecolor='black', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size})
    sns.heatmap(numeric_combined_matrix, mask=mask_lower, cmap='Blues', annot=combined_annotations, fmt="s", cbar=False,
                linewidths=.2, linecolor='black', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size})
    plt.title(my_title)
    plt.subplots_adjust(bottom=0.2)

    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    plt.yticks(rotation=0, fontsize=labels_font)
    plt.show()




file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_stats.csv'
df_full = pd.read_csv(file_path)


df_subset = df_full[0:16] 
process_and_visualize(df_subset, 'p-values and adjusted p-values - chunking y/n + chunk sizes', 7, 8)  


df_subset = df_full[17:47] 
process_and_visualize(df_subset, 'p-values and adjusted p-values - models', 7, 8)  


df_subset = df_full[48:203] 
process_and_visualize(df_subset, 'p-values and adjusted p-values - creativity params', 3, 6)  


df_subset = df_full[203:] 
process_and_visualize(df_subset, 'p-values and adjusted p-values - sync vs async + reproducibility + robustness', 4, 8)  



# color scheme based on adj p-values rather than p-values
# change to dark:low light:high
# change to bins: 0-0.05; 0.05-0.2; 0.2-0.6; 0.6-0.1
# one legend 







