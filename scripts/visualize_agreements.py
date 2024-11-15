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


def prepare_data(df):
    df = df.copy()
    column_rename_map = {
        'biome_exact_match_mean': 'exact_match',
        'biome_exact_match_sd': 'exact_match_sd',
        'biome_lenient_match_mean': 'lenient_match',
        'biome_lenient_match_sd': 'lenient_match_sd',
        'Average Similarity': 'avg_sim',
        'Standard Deviation': 'SD',
        'subbiome_sample_size': 'subbiome_size',
        'sample_size': 'biome_size'
    }
    df.rename(columns=column_rename_map, inplace=True)

    df['avg_sim'] *= 100
    df['SD'] *= 100

    df['exact_match_err'] = (df['exact_match_sd'] / np.sqrt(df['biome_size'])) 
    df['lenient_match_err'] = (df['lenient_match_sd'] / np.sqrt(df['biome_size']))
    df['avg_sim_err'] = df['SD'] / np.sqrt(df['subbiome_size'])
    return df

def plot_bars(df, plot_title, labels_font):
    df = prepare_data(df)
    
    n_points = len(df)
    width = max(0.15, 0.6 - 0.05 * n_points)  # Adjust bar width
    capsize = max(2, 10 - 0.8 * n_points)  # Adjust capsize
    
    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(n_points)
    
    ax.bar(indices - width, df['exact_match'], yerr=df['exact_match_err'], width=width, capsize=capsize, label='Exact Match')
    ax.bar(indices, df['lenient_match'], yerr=df['lenient_match_err'], width=width, capsize=capsize, label='Lenient Match')
    ax.bar(indices + width, df['avg_sim'], yerr=df['avg_sim_err'], width=width, capsize=capsize, label='Average Similarity')

    ax.set_xticks(indices)
    ax.set_xticklabels(df['Label'], rotation=90)
    plt.xlabel('')
    plt.ylabel('Scores (%)', fontsize=labels_font)
    plt.title(plot_title, fontsize=labels_font)
    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    
    ax.tick_params(axis='y', labelsize=labels_font) 

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_lines(df, plot_title, labels_font):
    df = prepare_data(df)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(len(df))

    ax.errorbar(indices, df['exact_match'], yerr=df['exact_match_err'], fmt='o-', capsize=5, label='Exact Match')
    ax.errorbar(indices, df['lenient_match'], yerr=df['lenient_match_err'], fmt='s-', capsize=5, label='Lenient Match')
    ax.errorbar(indices, df['avg_sim'], yerr=df['avg_sim_err'], fmt='^-', capsize=5, label='Average Similarity')

    ax.set_xticks(indices)
    ax.set_xticklabels(df['Label'], rotation=0)  # Set initial rotation to 0 for clarity

    plt.xlabel('')
    plt.ylabel('Scores (%)', fontsize=labels_font)
    plt.title(plot_title, fontsize=labels_font)

    # Use plt.xticks to set rotation, font size, and horizontal alignment
    plt.xticks(indices, df['Label'], rotation=45, fontsize=labels_font, ha='right')

    ax.tick_params(axis='y', labelsize=labels_font)  # Set y-axis font size without rotation

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()



def plot_lines_legend(df, labels_font):
    df = prepare_data(df)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(len(df))

    ax.errorbar(indices, df['exact_match'], yerr=df['exact_match_err'], fmt='o-', capsize=5, label='Exact Match')
    ax.errorbar(indices, df['lenient_match'], yerr=df['lenient_match_err'], fmt='s-', capsize=5, label='Lenient Match')
    ax.errorbar(indices, df['avg_sim'], yerr=df['avg_sim_err'], fmt='^-', capsize=5, label='Average Similarity')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = labels_font)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    
    
def plot_bars_legend(df, labels_font):
    df = prepare_data(df)
    
    n_points = len(df)
    width = max(0.15, 0.6 - 0.05 * n_points)  # Adjust bar width
    capsize = max(2, 10 - 0.8 * n_points)  # Adjust capsize
    
    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(n_points)
    
    ax.bar(indices - width, df['exact_match'], yerr=df['exact_match_err'], width=width, capsize=capsize, label='Exact Match')
    ax.bar(indices, df['lenient_match'], yerr=df['lenient_match_err'], width=width, capsize=capsize, label='Lenient Match')
    ax.bar(indices + width, df['avg_sim'], yerr=df['avg_sim_err'], width=width, capsize=capsize, label='Average Similarity')
    
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = labels_font)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()



file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_results.csv'
df_full = pd.read_csv(file_path)

df_subset = df_full[0:6] 
plot_lines(df_subset, 'Chunking (y/n) and chunk sizes (sync requests)', 9)

# Legends 
plot_bars_legend(df_subset, 9)
plot_lines_legend(df_subset, 9)


df_subset = df_full[7:13] 
plot_bars(df_subset, 'Models (sync requests)', 9)  










df_subset = df_full[14:33]  
plot_lines(df_subset, 'Creativity parameters (sync requests chunking)', 9)  

df_subset = df_full[34:53]  
plot_lines(df_subset, 'Creativity parameters (sync requests not chuking)', 9)  

df_subset = df_full[54:73] 
plot_lines(df_subset, 'Creativity parameters (async requests)', 9)  



# =============================================================================
# 
# df_subset = df_full[66:] 
# plot_lines(df_subset, 'Sync and async requests: different sample groups (rs)', 9)  
# 
# df_subset = df_full[115:128] 
# plot_bars(df_subset, 'Sync and async requests: same sample groups (rs)', 9)  
# 
# 
# df_subset = df_full[128:137] 
# plot_bars(df_subset, 'Please', 9)  
# 
# 
# df_subset = df_full[137:] 
# plot_bars(df_subset, 'Output formats', 9)  
# =============================================================================





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
    colors = sns.color_palette("Greens_r", n_colors=len(bins))  # Colors for biome
    colors_sub = sns.color_palette("Blues_r", n_colors=len(bins))  # Colors for sub-biome
    cmap_biome = LinearSegmentedColormap.from_list("custom_greens", colors, N=256)
    cmap_subbiome = LinearSegmentedColormap.from_list("custom_blues", colors_sub, N=256)
    #norm = BoundaryNorm(bins, ncolors=256, clip=True)

    plt.figure(figsize=(5.5, 5.5)) # width, height
    sns.heatmap(numeric_combined_matrix, mask=mask_upper, cmap=cmap_subbiome, annot=combined_annotations, fmt="s", cbar=False,
                linewidths=.2, linecolor='black', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size})
    sns.heatmap(numeric_combined_matrix, mask=mask_lower, cmap=cmap_biome, annot=combined_annotations, fmt="s", cbar=False,
                linewidths=.2, linecolor='black', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size})
    plt.title(my_title, fontsize=9)
    #plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.95, right=0.98, bottom=0.25)


    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    plt.yticks(rotation=0, fontsize=labels_font)
    plt.show()

    

    


file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/biome_subbiome_stats.csv'
df_full = pd.read_csv(file_path)

df_subset = df_full[0:29] 
process_and_visualize(df_subset, 'Chunking (y/n) and chunk sizes (sync requests)', 7, 8)  


df_subset = df_full[30:61] 
process_and_visualize(df_subset, 'Models (sync requests)', 7, 8)  

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
process_and_visualize(df_subset, 'Output formats', 8, 9)  





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






