#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:07:51 2024

@author: dgaio
"""


import matplotlib.pyplot as plt
import pandas as pd
import os


def lenient_match(a, b):
    if pd.isna(a) or pd.isna(b):
        return False  # No match if either value is missing
    if str(a) in str(b) or str(b) in str(a):
        if str(a) != str(b):  # Check if it's a partial match
            print(f"Partial match: a={a}, b={b}")
        return True
    return False


def plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir):
    # Compute full agreement statistics
    full_true_counts = full_agreement_df.groupby('label')['agreement'].sum()
    full_total_counts = full_agreement_df.groupby('label').size()
    full_result = pd.DataFrame({
        'Full True Counts': full_true_counts,
        'Full Total Counts': full_total_counts,
        'full match': (full_true_counts / full_total_counts * 100).round(2),
        'full match label': (full_true_counts / full_total_counts * 100).round(2).astype(str) + '%\n(n=' + full_total_counts.astype(str) + ')'
    })

    # Compute lenient agreement statistics
    lenient_true_counts = lenient_agreement_df.groupby('label')['agreement'].sum()
    lenient_total_counts = lenient_agreement_df.groupby('label').size()
    lenient_result = pd.DataFrame({
        'Lenient True Counts': lenient_true_counts,
        'Lenient Total Counts': lenient_total_counts,
        'full+partial match': (lenient_true_counts / lenient_total_counts * 100).round(2),
        'full+partial match label': (lenient_true_counts / lenient_total_counts * 100).round(2).astype(str) + '%\n(n=' + lenient_total_counts.astype(str) + ')'
    })

    # Merge results for plotting
    result = pd.merge(full_result, lenient_result, left_index=True, right_index=True, suffixes=('_full', '_lenient'))

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    indices = range(len(result))
    
    # Plot bars and annotate with pre-formatted labels
    for i, label in enumerate(result.index):
        bar1 = ax.bar(i - bar_width/2, result.at[label, 'full match'], bar_width, color='green', label='Full Match' if i == 0 else "")
        bar2 = ax.bar(i + bar_width/2, result.at[label, 'full+partial match'], bar_width, color='yellow', label='Full+Partial Match' if i == 0 else "")
        
        # Annotate bars with the pre-formatted labels
        ax.text(i - bar_width/2, bar1[0].get_height() + 0.5, result.at[label, 'full match label'], ha='center')
        ax.text(i + bar_width/2, bar2[0].get_height() + 0.5, result.at[label, 'full+partial match label'], ha='center')
    
    plt.title('Percentage of correct GPT output')
    plt.ylabel('Agreement (%)')
    plt.xlabel('Distinguishing Feature(s)')
    plt.xticks(indices, result.index, rotation=45)

    plt.tight_layout()

    # Adjust legend position
    plt.legend(title='', loc='upper right', title_fontsize='13', fontsize='11')

    # Constructing file name from unique parts of labels
    unique_parts = set(sum((label.split(', ') for label in file_label_map.values()), []))
    feature_description = "_".join(sorted(unique_parts))
    plot_filename = os.path.join(work_dir, f'agreement_{feature_description}.png')
    plt.savefig(plot_filename)
    plt.show()
    print(f"Plot saved as: {plot_filename}")
    



