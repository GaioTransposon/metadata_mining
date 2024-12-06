#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:02:23 2024

@author: dgaio
"""


import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from datetime import datetime
import time




home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")
plots_dir = os.path.join(work_dir, "plots")





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




def plot_bars(df, plots_dir, plot_title, labels_font):
    df = prepare_data(df)
    
    n_points = len(df)
    group_count = 3  # Since you have three bar groups per index
    
    # Calculate the total width for all bars in a group to be less than 1 unit
    total_group_width = 0.9  # Total width per group that bars can occupy
    width = total_group_width / group_count  # Individual bar width
    
    fig, ax = plt.subplots(figsize=(4, 4))  # Increased figure width for clarity
    indices = np.arange(n_points)
    
    # Calculate offset to center bars around the index point
    # Each bar is positioned so that the group is centered on the index
    offset = width * (group_count / 2 - 0.5)  # This offsets each group around its central index
    
    ax.bar(indices - offset, df['exact_match'], yerr=df['exact_match_err'], width=width, label='Exact Match', color='#265CA4')
    ax.bar(indices - offset + width, df['lenient_match'], yerr=df['lenient_match_err'], width=width, label='Lenient Match', color='#e37222')
    ax.bar(indices - offset + 2 * width, df['avg_sim'], yerr=df['avg_sim_err'], width=width, label='Average Similarity', color='#347734')
    
    ax.set_xticks(indices)
    ax.set_xticklabels(df['Label'], rotation=90)
    plt.xlabel('')
    plt.ylabel('Scores (%)', fontsize=labels_font)
    plt.title(plot_title, fontsize=labels_font)
    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    
    ax.tick_params(axis='y', labelsize=labels_font) 
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.show()
    
    # Save the figure to a PDF file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = os.path.join(plots_dir, f"results_plot_{current_time}.pdf")
    fig.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")
    time.sleep(1)  





def plot_lines(df, plots_dir, plot_title, labels_font, break_every_n=None):
    df = prepare_data(df)
    # Remove rows where 'Label' is NaN
    df = df.dropna(subset=['Label'])

    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(len(df))

    # Colors and formats for each metric
    metrics_info = {
        'Exact Match': {'color': '#265CA4', 'fmt': 'o-', 'markersize': None},
        'Lenient Match': {'color': '#e37222', 'fmt': 's-', 'markersize': 3},
        'Average Similarity': {'color': '#347734', 'fmt': '^-', 'markersize': None}
    }

    # Function to plot segments with discrete breaks
    def plot_segment(data, indices, color, fmt, capsize, markersize, label):
        ax.errorbar(indices, data[metric], yerr=data[metric + '_err'], fmt=fmt, capsize=capsize,
                    markersize=markersize, color=color, label=label if i == 0 else "")

    if break_every_n is None:
        break_every_n = len(df)  # Use entire length if no break is specified

    for metric, details in zip(['exact_match', 'lenient_match', 'avg_sim'], metrics_info.values()):
        for i in range(0, len(df), break_every_n):
            segment_indices = indices[i:i+break_every_n]
            segment_data = df.iloc[i:i+break_every_n]
            plot_segment(segment_data, segment_indices, details['color'], details['fmt'], 5, 
                         details['markersize'], metric)

    ax.set_xticks(indices)
    ax.set_xticklabels(df['Label'], rotation=0)
    plt.xlabel('')
    plt.ylabel('Scores (%)', fontsize=labels_font)
    plt.title(plot_title, fontsize=labels_font)
    plt.xticks(indices, df['Label'], rotation=45, fontsize=labels_font, ha='right')
    ax.tick_params(axis='y', labelsize=labels_font)
    ax.set_facecolor('white')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = os.path.join(plots_dir, f"results_plot_{current_time}.pdf")
    fig.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")
    time.sleep(1)  



def plot_lines_legend(df, plots_dir, labels_font):
    df = prepare_data(df)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(len(df))

    # Explicit colors for each metric
    colors = {
        'Exact Match': '#265CA4',  # Blue
        'Lenient Match': '#e37222',  # Orange
        'Average Similarity': '#347734'  # Green
    }

    # Exact Match
    ax.errorbar(indices, df['exact_match'], yerr=df['exact_match_err'], fmt='o-', capsize=5, color=colors['Exact Match'], label='Exact Match')
    
    # Lenient Match with smaller markersize
    ax.errorbar(indices, df['lenient_match'], yerr=df['lenient_match_err'], fmt='s-', capsize=5, markersize=3, color=colors['Lenient Match'], label='Lenient Match')  # Reduced markersize to 3
    
    # Average Similarity
    ax.errorbar(indices, df['avg_sim'], yerr=df['avg_sim_err'], fmt='^-', capsize=5, color=colors['Average Similarity'], label='Average Similarity')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=labels_font)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    
    # Save the figure to a PDF file
    pdf_filename = os.path.join(plots_dir, "results_legend_lines.pdf")
    fig.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")

    
    
def plot_bars_legend(df, plots_dir, labels_font):
    df = prepare_data(df)
    
    n_points = len(df)
    width = max(0.15, 0.6 - 0.05 * n_points)  # Adjust bar width
    capsize = max(2, 10 - 0.8 * n_points)  # Adjust capsize
    
    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(n_points)
    
    # Define colors for each metric
    colors = {
        'Exact Match': '#265CA4',  # Blue
        'Lenient Match': '#e37222',  # Orange
        'Average Similarity': '#347734'  # Green
    }
    
    ax.bar(indices - width, df['exact_match'], yerr=df['exact_match_err'], width=width, capsize=capsize, label='Exact Match', color=colors['Exact Match'])
    ax.bar(indices, df['lenient_match'], yerr=df['lenient_match_err'], width=width, capsize=capsize, label='Lenient Match', color=colors['Lenient Match'])
    ax.bar(indices + width, df['avg_sim'], yerr=df['avg_sim_err'], width=width, capsize=capsize, label='Average Similarity', color=colors['Average Similarity'])
    
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = labels_font)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    
    # Save the figure to a PDF file
    pdf_filename = os.path.join(plots_dir, "results_legend_bars.pdf")
    fig.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")
    



file_path = os.path.join(work_dir, "biome_subbiome_results.csv")
df_full = pd.read_csv(file_path)


###
# Figure 3
df_subset = df_full[0:6] 
plot_lines(df_subset, plots_dir, 'Effect of chunking (sync requests)', 9)
###

# Legends 
plot_bars_legend(df_subset, plots_dir, 9)
plot_lines_legend(df_subset, plots_dir, 9)

###
# Figure 4
df_subset = df_full[7:13] 
plot_bars(df_subset, plots_dir, 'Models (sync requests)', 9)  
###


###
# Figure 5 
df_subset = df_full[44:54]  
plot_lines(df_subset, plots_dir, 'Frequency penalty', 8, 4)  
###


###
# Suppl figure 
df_subset = df_full[14:28]  
plot_lines(df_subset, plots_dir, 'Creativity params: temp', 8, 4)  


df_subset = df_full[28:43]  
plot_lines(df_subset, plots_dir, 'Creativity params: topp', 8, 4)  


df_subset = df_full[44:59]  
plot_lines(df_subset, plots_dir, 'Creativity params: freqp', 8, 4)  

df_subset = df_full[59:74]  
plot_lines(df_subset, plots_dir, 'Creativity params: presp', 8, 4)  
###


###
# Suppl figure 
df_subset = df_full[75:91] 
plot_bars(df_subset, plots_dir, 'Sync requests: same and different sample groups (rs)', 7)  


df_subset = df_full[92:113] 
plot_bars(df_subset, plots_dir, 'Async requests: same and different sample groups (rs)', 7)  


df_subset = df_full[114:126] 
plot_bars(df_subset, plots_dir, 'Sync versus async requests (same sample group)', 7)  
###

# df_subset = df_full[127:135] 
# plot_bars(df_subset, plots_dir, 'Please', 9)  

###
# Figure 6
df_subset = df_full[136:] 
plot_bars(df_subset, plots_dir, 'Output formats', 9)  
###




################################################################################
################################################################################
################################################################################



# Stats:

def process_and_visualize(df, plots_dir, my_title, cell_font_size, labels_font):
    df = df.dropna()
    columns_to_keep = ['Label1', 'Label2', 'P-value', 'Adjusted P-value', 'Test Type', 'validation']
    df = df[columns_to_keep]
    labels = pd.unique(df[['Label1', 'Label2']].to_numpy().flatten())
    biome_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    subbiome_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    biome_annot = pd.DataFrame("", index=labels, columns=labels)
    subbiome_annot = pd.DataFrame("", index=labels, columns=labels)

    for _, row in df.iterrows():
        label1, label2 = row['Label1'], row['Label2']
        adj_p_value = row['Adjusted P-value']
        annotation = f"{row['P-value']:.2f};\n{adj_p_value:.2f}"
        if row['validation'] == 'biome':
            biome_matrix.at[label1, label2] = adj_p_value
            biome_matrix.at[label2, label1] = adj_p_value
            biome_annot.at[label1, label2] = annotation
            biome_annot.at[label2, label1] = annotation
        elif row['validation'] == 'sub-biome':
            subbiome_matrix.at[label1, label2] = adj_p_value
            subbiome_matrix.at[label2, label1] = adj_p_value
            subbiome_annot.at[label1, label2] = annotation
            subbiome_annot.at[label2, label1] = annotation

    # Prepare color bins and colormaps
    bins = [0, 0.01, 0.05, 0.2, 1.0]
    biome_colors = sns.color_palette("Blues_r", n_colors=len(bins)-1)
    subbiome_colors = sns.color_palette("Greens_r", n_colors=len(bins)-1)
    biome_cmap = LinearSegmentedColormap.from_list("BiomeCmap", biome_colors, N=len(bins)-1)
    subbiome_cmap = LinearSegmentedColormap.from_list("SubBiomeCmap", subbiome_colors, N=len(bins)-1)
    norm = BoundaryNorm(bins, ncolors=len(bins)-1, clip=True)

    plt.figure(figsize=(5.5, 5.5))
    # Use separate heatmaps with masks for biome and sub-biome data
    mask_upper = np.triu(np.ones_like(biome_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(biome_matrix, dtype=bool), k=-1)
    sns.heatmap(biome_matrix, mask=~mask_lower, cmap=biome_cmap, annot=biome_annot, fmt="s", cbar=False,
                linewidths=.5, linecolor='grey', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size}, norm=norm)
    sns.heatmap(subbiome_matrix, mask=~mask_upper, cmap=subbiome_cmap, annot=subbiome_annot, fmt="s", cbar=False,
                linewidths=.5, linecolor='grey', xticklabels=labels, yticklabels=labels, square=True,
                annot_kws={"size": cell_font_size}, norm=norm)
    plt.title(my_title, fontsize=labels_font)
    plt.subplots_adjust(top=0.95, right=0.98, bottom=0.25)
    plt.xticks(rotation=45, ha='right', fontsize=labels_font)
    plt.yticks(rotation=0, fontsize=labels_font)
    plt.show()
    
    # Save the figure to a PDF file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = os.path.join(plots_dir, f"stats_plot_{current_time}.pdf")
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")
    time.sleep(1)  







file_path = os.path.join(work_dir, "biome_subbiome_stats.csv")
df_full = pd.read_csv(file_path)




df_subset = df_full[0:30] 
process_and_visualize(df_subset, plots_dir, 'Effect of chunking (sync requests)', 8, 9)  


df_subset = df_full[30:61] 
process_and_visualize(df_subset, plots_dir, 'Models (sync requests)', 8, 9)  




df_subset = df_full[140:165] 
process_and_visualize(df_subset, plots_dir, 'Frequency penalty', 8, 9)  


df_subset = df_full[62:100] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: temp', 8, 9) 


df_subset = df_full[100:139] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: topp', 8, 9) 


df_subset = df_full[140:178] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: freqp', 8, 9)  


df_subset = df_full[179:217] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: presp', 8, 9)  


df_subset = df_full[218:490] 
process_and_visualize(df_subset, plots_dir, 'Sync requests: same and different sample groups (rs)', 4, 8)  

df_subset = df_full[491:911] 
process_and_visualize(df_subset, plots_dir, 'Async requests: same and different sample groups (rs)', 4, 8)  


df_subset = df_full[912:1044] 
process_and_visualize(df_subset, plots_dir, 'Sync versus async requests (same sample group)', 4, 8)  



# df_subset = df_full[1044:1101] 
# process_and_visualize(df_subset, plots_dir, 'Please (async)', 8, 9)  


df_subset = df_full[1101:] 
process_and_visualize(df_subset, plots_dir, 'Output formats', 8, 9) 



# Legend:
def create_pvalue_legend_with_ranges(bins, color_palette_1, color_palette_2, plots_dir, title):
    
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
    
    # Save the figure to a PDF file
    pdf_filename = os.path.join(plots_dir, "stats_legend.pdf")
    fig.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")

bins = [0, 0.01, 0.05, 0.2, 1.0]
create_pvalue_legend_with_ranges(bins, 'Greens_r', 'Blues_r', plots_dir, 'Adjusted p-values ranges')






