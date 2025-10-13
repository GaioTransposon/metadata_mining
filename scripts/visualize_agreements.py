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
from matplotlib.colors import ListedColormap, BoundaryNorm
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



#new
def plot_bars(df, plots_dir, plot_title, labels_font, order=None):
    df = prepare_data(df)

    # ---- Label index helper print (before any reordering)
    labels = df['Label'].tolist()
    print("\n[INFO] Label indices:")
    for i, lbl in enumerate(labels):
        print(f"  {i:>2}: {lbl}")

    # ---- Optional numeric reordering
    if order is not None:
        if not all(isinstance(i, (int, np.integer)) for i in order):
            raise ValueError("`order` must be a list of integer indices.")
        if not all(0 <= i < len(labels) for i in order):
            raise ValueError(f"`order` indices must be between 0 and {len(labels)-1}.")
        df = df.iloc[order]
        labels = [labels[i] for i in order]

        print("\n[INFO] Final plot order:")
        for i, lbl in enumerate(labels):
            print(f"  {i}: {lbl}")

    n_points = len(df)
    group_count = 3
    total_group_width = 0.9
    width = total_group_width / group_count

    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(n_points)
    offset = width * (group_count / 2 - 0.5)

    ax.bar(indices - offset, df['exact_match'], yerr=df['exact_match_err'], width=width, label='Exact Match', color='#265CA4')
    ax.bar(indices - offset + width, df['lenient_match'], yerr=df['lenient_match_err'], width=width, label='Lenient Match', color='#e37222')
    ax.bar(indices - offset + 2 * width, df['avg_sim'], yerr=df['avg_sim_err'], width=width, label='Average Similarity', color='#347734')

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=90)
    plt.xlabel('')
    plt.ylabel('Scores (%)', fontsize=labels_font)
    plt.title(plot_title, fontsize=labels_font)
    plt.xticks(rotation=45, fontsize=labels_font, ha='right')
    ax.tick_params(axis='y', labelsize=labels_font)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = os.path.join(plots_dir, f"results_plot_{current_time}.pdf")
    fig.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")
    time.sleep(1)



def plot_lines(df, plots_dir, plot_title, labels_font, break_every_n=None, order=None):
    df = prepare_data(df)
    df = df.dropna(subset=['Label'])

    # ---- Label index helper print (before any reordering)
    labels = df['Label'].tolist()
    print("\n[INFO] Label indices:")
    for i, lbl in enumerate(labels):
        print(f"  {i:>2}: {lbl}")

    # ---- Optional numeric reordering
    if order is not None:
        if not all(isinstance(i, (int, np.integer)) for i in order):
            raise ValueError("`order` must be a list of integer indices.")
        if not all(0 <= i < len(labels) for i in order):
            raise ValueError(f"`order` indices must be between 0 and {len(labels)-1}.")
        df = df.iloc[order]
        labels = [labels[i] for i in order]

        print("\n[INFO] Final plot order:")
        for i, lbl in enumerate(labels):
            print(f"  {i}: {lbl}")

    fig, ax = plt.subplots(figsize=(4, 4))
    indices = np.arange(len(df))

    metrics_info = {
        'Exact Match': {'color': '#265CA4', 'fmt': 'o-', 'markersize': None},
        'Lenient Match': {'color': '#e37222', 'fmt': 's-', 'markersize': 3},
        'Average Similarity': {'color': '#347734', 'fmt': '^-', 'markersize': None}
    }

    def plot_segment(data, indices, color, fmt, capsize, markersize, label):
        ax.errorbar(indices, data[metric], yerr=data[metric + '_err'],
                    fmt=fmt, capsize=capsize, markersize=markersize,
                    color=color, label=label if i == 0 else "")

    if break_every_n is None:
        break_every_n = len(df)

    for metric, details in zip(['exact_match', 'lenient_match', 'avg_sim'], metrics_info.values()):
        for i in range(0, len(df), break_every_n):
            segment_indices = indices[i:i+break_every_n]
            segment_data = df.iloc[i:i+break_every_n]
            plot_segment(segment_data, segment_indices, details['color'], details['fmt'], 5,
                         details['markersize'], metric)

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=0)
    plt.xlabel('')
    plt.ylabel('Scores (%)', fontsize=labels_font)
    plt.title(plot_title, fontsize=labels_font)
    plt.xticks(indices, labels, rotation=45, fontsize=labels_font, ha='right')
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
    

# Other possibility:
# file_path = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-8B.csv")

file_path = os.path.join(work_dir, "biome_subbiome_results__text-embedding-3-small.csv")
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
df_subset = df_full[136:140]  
plot_bars(df_subset, plots_dir, 'Output formats', 9)  
###



###
# Figure 6 
df_subset = df_full[44:54]  
plot_lines(df_subset, plots_dir, 'Frequency penalty', 8, 4)  
###



###
# Suppl Figure 5 - A
df_subset = df_full[14:28]  
plot_lines(df_subset, plots_dir, 'Creativity params: temp', 8, 4)  


# Suppl Figure 5 - B
df_subset = df_full[28:43]  
plot_lines(df_subset, plots_dir, 'Creativity params: topp', 8, 4)  


# Suppl Figure 5 - C
df_subset = df_full[44:59]  
plot_lines(df_subset, plots_dir, 'Creativity params: freqp', 8, 4)  


# Suppl Figure 5 - D
df_subset = df_full[59:74]  
plot_lines(df_subset, plots_dir, 'Creativity params: presp', 8, 4)  
###


###
# Suppl figure 6 - A
df_subset = df_full[75:91] 
plot_bars(df_subset, plots_dir, 'Sync requests: same and different sample groups (rs)', 7)  

# Suppl figure 6 - B
df_subset = df_full[92:113] 
plot_bars(df_subset, plots_dir, 'Async requests: same and different sample groups (rs)', 7)  

# Suppl figure 6 - C
df_subset = df_full[114:126] 
plot_bars(df_subset, plots_dir, 'Async versus sync requests (same sample group)', 7)  
###


# df_subset = df_full[127:135] 
# plot_bars(df_subset, plots_dir, 'Please', 9)  




###
# Figure 8
file_path = os.path.join(work_dir, "biome_subbiome_results__text-embedding-3-small.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[148:153] 
plot_bars(df_subset, plots_dir, 'text-embedding-3-small', 7, order=[2, 0, 1, 4, 3])  


file_path = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-0.6B.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[148:153] 
plot_bars(df_subset, plots_dir, 'Qwen-Qwen3-Embedding-0.6B', 7, order=[2, 0, 1, 4, 3])  


file_path = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-4B.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[148:153] 
plot_bars(df_subset, plots_dir, 'Qwen-Qwen3-Embedding-4B', 7, order=[2, 0, 1, 4, 3])  


file_path = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-8B.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[148:153] 
plot_bars(df_subset, plots_dir, 'Qwen-Qwen3-Embedding-8B', 7, order=[2, 0, 1, 4, 3])  









################################################################################
################################################################################
################################################################################


# Stats:





def process_and_visualize(
        df,
        plots_dir,
        my_title,
        cell_font_size=8,
        labels_font=9,
        debug=True,
        order=None):   # optional numeric order
    """
    Draw dual heat-maps for biome vs. sub-biome tests and save to PDF.
    Raises or warns early if something is wrong with the data.
    """

    # ---------- 1️⃣  Basic cleaning ------------------------------------------------
    df = df.copy()        # avoid SettingWithCopyWarning
    df = df.dropna()

    if df.empty:
        raise ValueError(
            "After dropna() the DataFrame is empty – nothing to plot. "
            "Check for missing values in 'Label1', 'Label2', 'P-value', etc."
        )

    cols_needed = ['Label1', 'Label2', 'P-value', 'Adjusted P-value',
                   'Test Type', 'validation']
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")

    # Make validation comparisons case- & space-insensitive
    df['validation'] = df['validation'].str.strip().str.lower()

    # ---------- 2️⃣  Quick stats printout -----------------------------------------
    biome_rows    = (df['validation'] == 'biome').sum()
    subbiome_rows = (df['validation'] == 'sub-biome').sum()
    if debug:
        print(f"[DEBUG] rows after dropna: {len(df)} "
              f"(biome={biome_rows}, sub-biome={subbiome_rows})")

    if biome_rows == 0 and subbiome_rows == 0:
        raise ValueError(
            "No rows have validation == 'biome' or 'sub-biome'. "
            "Check spelling/capitalisation in that column."
        )

    # ---------- 3️⃣  Build matrices ----------------------------------------------
    labels = list(pd.unique(df[['Label1', 'Label2']].to_numpy().flatten()))

    # Helper print: indices -> labels
    print("\n[INFO] Label indices:")
    for i, lbl in enumerate(labels):
        print(f"  {i:>2}: {lbl}")

    biome_matrix    = pd.DataFrame(np.nan, index=labels, columns=labels)
    subbiome_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    biome_annot     = pd.DataFrame("",     index=labels, columns=labels)
    subbiome_annot  = pd.DataFrame("",     index=labels, columns=labels)

    for _, row in df.iterrows():
        l1, l2 = row['Label1'], row['Label2']
        ann    = f"{row['P-value']:.2f};\n{row['Adjusted P-value']:.2f}"
        target = row['validation']
        if target == 'biome':
            biome_matrix.loc[l1, l2] = biome_matrix.loc[l2, l1] = row['Adjusted P-value']
            biome_annot .loc[l1, l2] = biome_annot .loc[l2, l1] = ann
        elif target == 'sub-biome':
            subbiome_matrix.loc[l1, l2] = subbiome_matrix.loc[l2, l1] = row['Adjusted P-value']
            subbiome_annot .loc[l1, l2] = subbiome_annot .loc[l2, l1] = ann

    if biome_matrix.isna().all().all() and subbiome_matrix.isna().all().all():
        raise ValueError(
            "Both matrices are still all-NaN after filling. "
            "Double-check Label1/Label2 pairs against the 'validation' column."
        )

    # --- reorder by numeric indices, if provided (after filling matrices) ---
    if order is not None:
        if not all(isinstance(i, (int, np.integer)) for i in order):
            raise ValueError("`order` must be a list of integer indices.")
        if not all(0 <= i < len(labels) for i in order):
            raise ValueError(f"`order` indices must be between 0 and {len(labels)-1}.")

        labels = [labels[i] for i in order]

        biome_matrix    = biome_matrix.loc[labels, labels]
        subbiome_matrix = subbiome_matrix.loc[labels, labels]
        biome_annot     = biome_annot.loc[labels, labels]
        subbiome_annot  = subbiome_annot.loc[labels, labels]

        print("\n[INFO] Final plot order:")
        for i, lbl in enumerate(labels):
            print(f"  {i}: {lbl}")

    # ---------- 4️⃣  Plot ---------------------------------------------------------
    bins = [0, 0.01, 0.05, 0.2, 1.0]

    # use DISCRETE colormaps with exactly len(bins)-1 colors
    biome_cmap    = ListedColormap(sns.color_palette("Blues_r",  len(bins)-1))
    subbiome_cmap = ListedColormap(sns.color_palette("Greens_r", len(bins)-1))

    # BoundaryNorm should use the cmap's number of discrete colors
    norm = BoundaryNorm(bins, biome_cmap.N, clip=True)

    fig = plt.figure(figsize=(5.5, 5.5))

    mask_upper = np.triu(np.ones_like(biome_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(biome_matrix, dtype=bool), k=-1)

    sns.heatmap(
        biome_matrix,   mask=~mask_lower, cmap=biome_cmap,
        annot=biome_annot, fmt="s", cbar=False,
        linewidths=.5, linecolor='grey', xticklabels=labels, yticklabels=labels,
        square=True, annot_kws={"size": cell_font_size}, norm=norm)

    sns.heatmap(
        subbiome_matrix, mask=~mask_upper, cmap=subbiome_cmap,
        annot=subbiome_annot, fmt="s", cbar=False,
        linewidths=.5, linecolor='grey', xticklabels=labels, yticklabels=labels,
        square=True, annot_kws={"size": cell_font_size}, norm=norm)

    plt.title(my_title, fontsize=labels_font)
    plt.xticks(rotation=45, ha='right', fontsize=labels_font)
    plt.yticks(rotation=0,  fontsize=labels_font)
    plt.tight_layout()

    # ---------- 5️⃣  Save BEFORE show() ------------------------------------------
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)

    pdf_name = os.path.join(plots_dir, f"stats_plot_{current_time}.pdf")
    fig.savefig(pdf_name, bbox_inches="tight")
    if debug:
        print(f"[DEBUG] plot written to: {pdf_name}")

    plt.show()
    plt.close(fig)
    time.sleep(1)



# biome_subbiome_stats__Qwen-Qwen3-Embedding-8B.csv

file_path = os.path.join(work_dir, "biome_subbiome_stats__text-embedding-3-small.csv")
df_full = pd.read_csv(file_path)



# Figure 3
df_subset = df_full[0:30] 
process_and_visualize(df_subset, plots_dir, 'Effect of chunking (sync requests)', 8, 9)  




file_path = os.path.join(work_dir, "biome_subbiome_stats.csv")
df_full = pd.read_csv(file_path)



# Figure 3
df_subset = df_full[0:30] 
process_and_visualize(df_subset, plots_dir, 'Effect of chunking (sync requests)', 8, 9)  

# Figure 4
df_subset = df_full[30:61] 
process_and_visualize(df_subset, plots_dir, 'Models (sync requests)', 8, 9)  


# Figure 5
df_subset = df_full[1101:1114] 
process_and_visualize(df_subset, plots_dir, 'Output formats', 8, 9) 


# Figure 6 
df_subset = df_full[140:165] 
process_and_visualize(df_subset, plots_dir, 'Frequency penalty', 8, 9)  




# Suppl Figure 5 - E
df_subset = df_full[62:100] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: temp', 8, 9) 

# Suppl Figure 5 - F
df_subset = df_full[100:139] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: topp', 8, 9) 

# Suppl Figure 5 - G
df_subset = df_full[140:178] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: freqp', 8, 9)  

# Suppl Figure 5 - H
df_subset = df_full[179:217] 
process_and_visualize(df_subset, plots_dir, 'Creativity parameters: presp', 8, 9)  


# Suppl Figure 6 - D
df_subset = df_full[218:490] 
process_and_visualize(df_subset, plots_dir, 'Sync requests: same and different sample groups (rs)', 4, 8)  

# Suppl Figure 6 - E            
df_subset = df_full[491:911] 
process_and_visualize(df_subset, plots_dir, 'Async requests: same and different sample groups (rs)', 4, 8)  

# Suppl Figure 6 - F
df_subset = df_full[912:1044] 
process_and_visualize(df_subset, plots_dir, 'Async versus sync requests (same sample group)', 4, 8)  




# df_subset = df_full[1044:1101] 
# process_and_visualize(df_subset, plots_dir, 'Please (async)', 8, 9)  



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





###
# Figure 8
file_path = os.path.join(work_dir, "biome_subbiome_stats__text-embedding-3-small.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[1129:] 
process_and_visualize(df_subset, plots_dir, 'text-embedding-3-small', 7, 9, order=[2, 4, 1, 0, 3])

file_path = os.path.join(work_dir, "biome_subbiome_stats__Qwen-Qwen3-Embedding-8B.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[1129:] 
process_and_visualize(df_subset, plots_dir, 'Qwen-Qwen3-Embedding-8B', 7, 9, order=[2, 4, 1, 0, 3])

file_path = os.path.join(work_dir, "biome_subbiome_stats__Qwen-Qwen3-Embedding-4B.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[1129:] 
process_and_visualize(df_subset, plots_dir, 'Qwen-Qwen3-Embedding-4B', 7, 9, order=[2, 4, 1, 0, 3])

file_path = os.path.join(work_dir, "biome_subbiome_stats__Qwen-Qwen3-Embedding-0.6B.csv")
df_full = pd.read_csv(file_path)
df_subset = df_full[1129:] 
process_and_visualize(df_subset, plots_dir, 'Qwen-Qwen3-Embedding-0.6B', 7, 9, order=[2, 4, 1, 0, 3])
###







################################################################################
################################################################################
################################################################################


# Figure 8:




# Figure 8:

# ==== inputs ====
file_A = os.path.join(work_dir, "biome_subbiome_results__text-embedding-3-small.csv")        # A
file_B = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-8B.csv")       # B
file_C = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-4B.csv")       # C
file_D = os.path.join(work_dir, "biome_subbiome_results__Qwen-Qwen3-Embedding-0.6B.csv")     # D

rows_of_interest = [146, 147, 148, 149, 150, 151, 152]  # same rows in all files
order = [4, 2, 3, 5, 1, 6, 0]                           # desired visual order
baseline_idx = 0  # used only for Panel B brackets (within-embedding vs baseline)

# ==== helpers ====
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt
from scipy.stats import t as student_t
from datetime import datetime

def welch_ttest_from_summary(m1, s1, n1, m2, s2, n2):
    se = sqrt(s1**2/n1 + s2**2/n2)
    if se == 0:
        return 1.0
    t = (m1 - m2) / se
    df_num = (s1**2/n1 + s2**2/n2)**2
    df_den = (s1**4 / (n1**2 * (n1 - 1))) + (s2**4 / (n2**2 * (n2 - 1)))
    df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)
    return 2 * (1 - student_t.cdf(abs(t), df))

def benjamini_hochberg(pvals, alpha=0.05):
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool)
    order_idx = np.argsort(pvals)
    ranked = np.array(pvals)[order_idx]
    thresh = (np.arange(1, m+1) / m) * alpha
    passed = ranked <= thresh
    if passed.any():
        max_k = np.max(np.where(passed)[0])
        passed[:max_k+1] = True
        passed[max_k+1:] = False
    mask = np.zeros(m, dtype=bool)
    mask[order_idx] = passed
    return mask

def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.0, color='#444444'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=color, linewidth=lw)
    ax.text((x1+x2)/2, y+h+0.35, text, ha='center', va='bottom', fontsize=8, color=color)

def pairwise_pvals_welch(means, sds, ns):
    """All pairwise Welch tests; returns pairs and p-values."""
    pairs, pvals = [], []
    k = len(means)
    for i, j in combinations(range(k), 2):
        p = welch_ttest_from_summary(means[i], sds[i], ns[i], means[j], sds[j], ns[j])
        pairs.append((i, j)); pvals.append(p)
    return pairs, np.array(pvals)

def cld_from_pvals(k, pairs, pvals, alpha=0.05):
    """
    Compact-letter display after BH-FDR.
    Items sharing a letter are NOT significantly different.
    """
    sig = benjamini_hochberg(pvals, alpha=alpha)
    diff = np.zeros((k, k), dtype=bool)  # True = significantly different
    for (i, j), s in zip(pairs, sig):
        diff[i, j] = diff[j, i] = s
    letters = [''] * k
    next_letter = ord('a')
    remaining = set(range(k))
    while remaining:
        group = []
        for i in list(remaining):
            if all(not diff[i, g] for g in group):
                group.append(i)
        for i in group:
            letters[i] += chr(next_letter)
        remaining -= set(group)
        next_letter += 1
    return letters

def sig_lines_vs_baseline(ax, means, sds, ns, color='#444444', y_pad=1.2, step=1.7, alpha=0.05, x=None):
    """Brackets only vs baseline index to avoid clutter (Panel B)."""
    if x is None:
        x = np.arange(len(means))
    base = baseline_idx
    pvals, pairs = [], []
    for j in range(len(means)):
        if j == base: 
            continue
        p = welch_ttest_from_summary(means[base], sds[base], ns[base], means[j], sds[j], ns[j])
        pvals.append(p); pairs.append((base, j))
    sig = benjamini_hochberg(pvals, alpha=alpha)
    level = 0
    for keep, (i, j) in zip(sig, pairs):
        if not keep: 
            continue
        top = max(means[i], means[j])
        y = top + y_pad + level*step
        add_sig_bracket(ax, x[i], x[j], y, 0.8, "*", color=color)
        level += 1

def within_group_sig_brackets(ax, x, offsets, means_list, sds_list, ns_list,
                              compare_indices=(1,2,3), alpha=0.05,
                              y_pad=0.6, step=0.9, color='#2f2f2f'):
    """
    Draw brackets within each x-position across selected embeddings.
    compare_indices: tuple of indices into means_list to include (default: Qwen B,C,D = 1,2,3).
    For each position, we test all pairwise combos among these indices; BH-FDR *within the position*.
    """
    idxs = list(compare_indices)
    pair_template = list(combinations(range(len(idxs)), 2))  # pairs over local indices
    for t in range(len(x)):
        # collect per-embedding numbers at this x-position
        vals  = [means_list[i][t] for i in idxs]
        sds   = [sds_list[i][t]   for i in idxs]
        ns    = [ns_list[i][t]    for i in idxs]

        # compute p-values for the local pairs
        pvals, pairs_abs = [], []
        for (a, b) in pair_template:
            i_abs, j_abs = idxs[a], idxs[b]
            p = welch_ttest_from_summary(vals[a], sds[a], ns[a], vals[b], sds[b], ns[b])
            pvals.append(p)
            pairs_abs.append((i_abs, j_abs))  # absolute indices within means_list

        sig_mask = benjamini_hochberg(pvals, alpha=alpha)
        level = 0
        for keep, (i_abs, j_abs) in zip(sig_mask, pairs_abs):
            if not keep:
                continue
            # bracket between bar centers for this x
            xi = x[t] + offsets[i_abs]
            xj = x[t] + offsets[j_abs]
            top = max(means_list[i_abs][t], means_list[j_abs][t])
            y  = top + y_pad + level*step
            add_sig_bracket(ax, xi, xj, y, 0.6, "*", color=color)
            level += 1

# ==== load & prep ====
dfA = prepare_data(pd.read_csv(file_A)).loc[rows_of_interest].reset_index(drop=True)
dfB = prepare_data(pd.read_csv(file_B)).loc[rows_of_interest].reset_index(drop=True)
dfC = prepare_data(pd.read_csv(file_C)).loc[rows_of_interest].reset_index(drop=True)
dfD = prepare_data(pd.read_csv(file_D)).loc[rows_of_interest].reset_index(drop=True)

labels = dfA['Label'].tolist()
dfA, dfB, dfC, dfD = (df.iloc[order].reset_index(drop=True) for df in (dfA, dfB, dfC, dfD))
labels = [labels[i] for i in order]
x = np.arange(len(labels))

# Biome (left panel) from file A
biome_exact, biome_exact_sd, biome_n = dfA['exact_match'], dfA['exact_match_sd'], dfA['biome_size']
biome_len,   biome_len_sd,   biome_n2 = dfA['lenient_match'], dfA['lenient_match_sd'], dfA['biome_size']

# Sub-biome (right panel) from 4 embeddings
sub_A, sub_A_sd, n_subA = dfA['avg_sim'], dfA['SD'], dfA['subbiome_size']
sub_B, sub_B_sd, n_subB = dfB['avg_sim'], dfB['SD'], dfB['subbiome_size']
sub_C, sub_C_sd, n_subC = dfC['avg_sim'], dfC['SD'], dfC['subbiome_size']
sub_D, sub_D_sd, n_subD = dfD['avg_sim'], dfD['SD'], dfD['subbiome_size']

# ==== styling ====
c_biomeE = '#265CA4'
c_biomeL = '#e37222'
c_mean   = '#9E9E9E'

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# === SIDE-BY-SIDE LAYOUT WITH SUB-BIOME Y-AXIS ZOOM ===
fig = plt.figure(figsize=(10.5, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.32)

# ---------- LEFT: Biome (Exact + Lenient) ----------
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(x, biome_exact, yerr=biome_exact_sd/np.sqrt(biome_n),
             fmt='o-', capsize=4, lw=1.2, color=c_biomeE, label='Biome (Exact)')
ax1.errorbar(x, biome_len,   yerr=biome_len_sd/np.sqrt(biome_n2),
             fmt='s--', capsize=4, lw=1.0, color=c_biomeL, label='Biome (Lenient)', markersize=4)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=28, ha='right')
ax1.set_ylabel('Scores (%)')
# (no overall title)

# compress y-range to highlight clustering
m = float(min(biome_exact.min(), biome_len.min()))
M = float(max(biome_exact.max(), biome_len.max()))
pad = max(1.0, (M - m) * 0.7)
ax1.set_ylim(max(0, m - pad), min(100, M + pad))

# mean line + value
mean_b = float(biome_exact.mean())
ax1.axhline(mean_b, linestyle=(0,(4,4)), linewidth=1, color=c_mean)
ax1.text(x[-1] + 0.25, mean_b, f"mean {mean_b:.1f}%", va='center', fontsize=9, color=c_mean)

# light grids & clean spines
for ax in (ax1,):
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax1.legend(loc='upper left', frameon=False, fontsize=9)

# ---- Compact Letter Display (all pairwise Welch tests, BH-FDR) ----
pairsE, pvalsE = pairwise_pvals_welch(biome_exact.values, biome_exact_sd.values, biome_n.values)
letters_exact  = cld_from_pvals(len(labels), pairsE, pvalsE, alpha=0.05)

pairsL, pvalsL = pairwise_pvals_welch(biome_len.values, biome_len_sd.values, biome_n2.values)
letters_len    = cld_from_pvals(len(labels), pairsL, pvalsL, alpha=0.05)

# place letters (Exact above Lenient)
y_exact = biome_exact.values + biome_exact_sd.values/np.sqrt(biome_n.values) + 0.6
y_len   = biome_len.values   + biome_len_sd.values/np.sqrt(biome_n2.values) + 0.6
for xi in range(len(labels)):
    ax1.text(xi, y_exact[xi] + 0.8, letters_exact[xi], ha='center', va='bottom',
             fontsize=8, color=c_biomeE, fontweight='bold')
    ax1.text(xi, y_len[xi] + 0.0,   letters_len[xi],   ha='center', va='bottom',
             fontsize=8, color=c_biomeL, fontweight='bold')

# ---------- RIGHT: Sub-biome (4 embeddings in green family) ----------
ax2 = fig.add_subplot(gs[0, 1])

# green gradient + hatches
c_A = '#347734'   # text-embedding-3-small
c_B = '#4C8C4A'   # Qwen-8B
c_C = '#6BA96C'   # Qwen-4B
c_D = '#A8D5A2'   # Qwen-0.6B
hatches = ['', '///', '\\\\', '...']

group_w = 0.80
w = group_w / 4
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]  # A, B, C, D

bar_sets = [
    dict(vals=sub_A, err=sub_A_sd/np.sqrt(n_subA), color=c_A, label='text-embedding-3-small'),
    dict(vals=sub_B, err=sub_B_sd/np.sqrt(n_subB), color=c_B, label='Qwen-Qwen3-Embedding-8B'),
    dict(vals=sub_C, err=sub_C_sd/np.sqrt(n_subC), color=c_C, label='Qwen-Qwen3-Embedding-4B'),
    dict(vals=sub_D, err=sub_D_sd/np.sqrt(n_subD), color=c_D, label='Qwen-Qwen3-Embedding-0.6B'),
]
for (off, b, hatch) in zip(offsets, bar_sets, hatches):
    ax2.bar(x + off, b['vals'], yerr=b['err'], width=w,
            label=b['label'], color=b['color'],
            edgecolor='#1B1B1B', linewidth=0.4, hatch=hatch)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=28, ha='right')
ax2.set_ylabel('Scores (%)')
ax2.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9,
           title='Sub-biome avg cosine similarity')

ax2.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# ---- y-axis zoom (cut lower portion for readability) ----
zoom_min = 50
ymax = float(np.max([sub_A.max(), sub_B.max(), sub_C.max(), sub_D.max()])) + 2.0
ax2.set_ylim(zoom_min, min(100, ymax))

# tiny break marks to indicate truncation
d = 0.012
kwargs = dict(transform=ax2.transAxes, color='#555555', clip_on=False, linewidth=1.0)
ax2.plot((-d, +d), (0, 0), **kwargs)           # bottom-left
ax2.plot((1 - d, 1 + d), (0, 0), **kwargs)     # bottom-right

# --- CLD letters for Sub-biome: text-embedding-3-small ONLY ---
pairs_subA, pvals_subA = pairwise_pvals_welch(sub_A.values, sub_A_sd.values, n_subA.values)
letters_subA = cld_from_pvals(len(labels), pairs_subA, pvals_subA, alpha=0.05)
y_subA = sub_A.values + sub_A_sd.values/np.sqrt(n_subA.values) + 0.6
for xi in range(len(labels)):
    ax2.text(x[xi] + offsets[0], y_subA[xi] + 0.5, letters_subA[xi],
             ha='center', va='bottom', fontsize=8, color=c_A, fontweight='bold')

# --- NEW: within-group significance (Qwen-only B/C/D) at each x ---
within_group_sig_brackets(
    ax2, x, offsets,
    means_list=[sub_A.values, sub_B.values, sub_C.values, sub_D.values],
    sds_list=[sub_A_sd.values, sub_B_sd.values, sub_C_sd.values, sub_D_sd.values],
    ns_list=[n_subA.values, n_subB.values, n_subC.values, n_subD.values],
    compare_indices=(1, 2, 3),    # only among Qwen-8B, Qwen-4B, Qwen-0.6B
    alpha=0.05,
    y_pad=0.6, step=0.9, color='#2f2f2f'
)

# ---- significance vs baseline within each embedding (kept minimal) ----
sig_lines_vs_baseline(ax2, sub_A.values, sub_A_sd.values, n_subA.values,
                      color=c_A, y_pad=0.8, step=1.2, alpha=0.05, x=x)
sig_lines_vs_baseline(ax2, sub_B.values, sub_B_sd.values, n_subB.values,
                      color=c_B, y_pad=1.9, step=1.2, alpha=0.05, x=x)
sig_lines_vs_baseline(ax2, sub_C.values, sub_C_sd.values, n_subC.values,
                      color=c_C, y_pad=3.0, step=1.2, alpha=0.05, x=x)
sig_lines_vs_baseline(ax2, sub_D.values, sub_D_sd.values, n_subD.values,
                      color=c_D, y_pad=4.1, step=1.2, alpha=0.05, x=x)

plt.tight_layout()

# save
out = os.path.join(plots_dir, f"figure7_side_by_side_zoom_cld_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
fig.savefig(out, bbox_inches='tight')
print(f"[INFO] Figure 7 saved as {out}")
plt.show()





# --- diagnostics: Qwen-within-label p-values (Panel B) ---
def qwen_within_label_diagnostics(label_idx, label_name):
    means = [sub_B.values[label_idx], sub_C.values[label_idx], sub_D.values[label_idx]]
    sds   = [sub_B_sd.values[label_idx], sub_C_sd.values[label_idx], sub_D_sd.values[label_idx]]
    ns    = [n_subB.values[label_idx], n_subC.values[label_idx], n_subD.values[label_idx]]
    names = ["Qwen-8B", "Qwen-4B", "Qwen-0.6B"]
    pairs, raw_p = [], []
    for (i, j) in [(0,1),(0,2),(1,2)]:
        p = welch_ttest_from_summary(means[i], sds[i], ns[i], means[j], sds[j], ns[j])
        pairs.append((names[i], names[j], means[j]-means[i]))
        raw_p.append(p)
    adj = benjamini_hochberg(raw_p, alpha=0.05)
    print(f"\n[Qwen comparisons @ {label_name}]")
    for (a,b,delta), p, keep in zip(pairs, raw_p, adj):
        star = "*" if keep else "ns"
        print(f"  {a} vs {b}: Δ={delta:+.2f}  p={p:.4g}  ({star} after BH)")
        
# print for all labels (or pick just 'gpt-4.1')
for idx, lab in enumerate(labels):
    qwen_within_label_diagnostics(idx, lab)
