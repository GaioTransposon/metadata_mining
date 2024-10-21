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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple


# Results:

def plot_data(df, plot_title, labels_font):
    
    
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
    plt.title(plot_title,fontsize=12)
    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=8) 
    
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()




file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_results.csv'
df_full = pd.read_csv(file_path)


df_subset = df_full[0:8] 
plot_data(df_subset, 'Chunking (y/n) and chunk sizes (sync requests)', 9)  

df_subset = df_full[9:15] 
plot_data(df_subset, 'Models (sync requests)', 9)  



df_subset = df_full[16:35]  
plot_data(df_subset, 'Creativity parameters (sync requests chunking)', 9)  

df_subset = df_full[35:55]  
plot_data(df_subset, 'Creativity parameters (sync requests not chuking)', 9)  

df_subset = df_full[55:75] 
plot_data(df_subset, 'Creativity parameters (async requests)', 9)  


df_subset = df_full[75:115] 
plot_data(df_subset, 'Sync and async requests: different sample groups (rs)', 9)  

df_subset = df_full[115:128] 
plot_data(df_subset, 'Sync and async requests: same sample groups (rs)', 9)  


df_subset = df_full[128:137] 
plot_data(df_subset, 'Please', 9)  


df_subset = df_full[137:] 
plot_data(df_subset, 'Output formats', 9)  



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
        adj_p_value = row['Adjusted P-value']
        validation_type = row['validation']
        annotation = f"{row['P-value']:.2f};\n{adj_p_value:.2f}"

        if validation_type == 'biome':
            biome_matrix.at[label1, label2] = adj_p_value
            biome_matrix.at[label2, label1] = adj_p_value
            biome_annot.at[label1, label2] = annotation
            biome_annot.at[label2, label1] = annotation

        elif validation_type == 'sub-biome':
            subbiome_matrix.at[label1, label2] = adj_p_value
            subbiome_matrix.at[label2, label1] = adj_p_value
            subbiome_annot.at[label1, label2] = annotation
            subbiome_annot.at[label2, label1] = annotation

    
    
    # Merge matrices
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

    bins = [0, 0.01, 0.05, 0.2, 1.0]
    colors = sns.color_palette("Blues_r", n_colors=len(bins))  # Colors for biome
    colors_sub = sns.color_palette("Greens_r", n_colors=len(bins))  # Colors for sub-biome
    cmap_biome = LinearSegmentedColormap.from_list("custom_blues", colors, N=256)
    cmap_subbiome = LinearSegmentedColormap.from_list("custom_greens", colors_sub, N=256)
    #norm = BoundaryNorm(bins, ncolors=256, clip=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_combined_matrix, mask=mask_upper, cmap=cmap_subbiome, annot=combined_annotations, fmt="s", cbar=False,
                linewidths=.2, linecolor='black', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size})
    sns.heatmap(numeric_combined_matrix, mask=mask_lower, cmap=cmap_biome, annot=combined_annotations, fmt="s", cbar=False,
                linewidths=.2, linecolor='black', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size})
    plt.title(my_title, fontsize=9)
    plt.subplots_adjust(bottom=0.2)

    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    plt.yticks(rotation=0, fontsize=labels_font)
    plt.show()

    

    


file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_stats.csv'
df_full = pd.read_csv(file_path)

df_subset = df_full[0:24] 
process_and_visualize(df_subset, 'Chunking (y/n) and chunk sizes (sync requests)', 8, 9)  


df_subset = df_full[23:54] 
process_and_visualize(df_subset, 'Models (sync requests)', 8, 9)  

df_subset = df_full[59:106] 
process_and_visualize(df_subset, 'Creativity parameters (sync requests chunking)', 8, 9) 

df_subset = df_full[106:159] 
process_and_visualize(df_subset, 'Creativity parameters (sync requests no chunking)', 8, 9) 

df_subset = df_full[159:210] 
process_and_visualize(df_subset, 'Creativity parameters (async requests)', 8, 9)  


df_subset = df_full[210:904] 
process_and_visualize(df_subset, 'Sync and async requests: different sample groups (rs)', 4, 8)  


df_subset = df_full[904:1037] 
process_and_visualize(df_subset, 'Sync and async requests: same sample groups (rs)', 4, 8)  


df_subset = df_full[1037:1094] 
process_and_visualize(df_subset, 'Please', 4, 8)  

df_subset = df_full[1094:] 
process_and_visualize(df_subset, 'Output format', 4, 8)  





# Legend:
def create_pvalue_legend_with_ranges(bins, color_palette_1, color_palette_2, title):
    
    # create single subplot
    fig, ax = plt.subplots(figsize=(6, 2))

    colors_1 = sns.color_palette(color_palette_1, n_colors=len(bins))
    colors_2 = sns.color_palette(color_palette_2, n_colors=len(bins))

    # define bins and corresponding labels
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

    handles = []
    for i, label in enumerate(bin_labels):
        patch1 = plt.Rectangle((0,0), 1, 1, facecolor=colors_1[i])
        patch2 = plt.Rectangle((0,0), 1, 1, facecolor=colors_2[i])
        handle = (patch1, patch2)
        handles.append((handle, label))

    ax.legend([handle for handle, label in handles], [label for handle, label in handles],
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, title=title, loc='upper center', 
                       frameon=True, bbox_to_anchor=(0.5, 1), handlelength=3, handletextpad=1)
    
    ax.set_title(title, pad=15)
    ax.axis('off')  

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

bins = [0, 0.01, 0.05, 0.2, 1.0]
create_pvalue_legend_with_ranges(bins, 'Greens_r', 'Blues_r', 'Adjusted p-values ranges')








