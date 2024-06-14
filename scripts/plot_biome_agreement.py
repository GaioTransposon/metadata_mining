#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:07:51 2024

@author: dgaio
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_biome_agreement(concatenated_df, file_label_map, work_dir):
    # Compute agreement statistics
    true_counts = concatenated_df.groupby('label')['agreement'].sum()
    total_counts = concatenated_df.groupby('label').size()
    result = pd.DataFrame({
        'True Counts': true_counts,
        'Total Counts': total_counts,
        'Percentage True': (true_counts / total_counts * 100).round(2)
    })

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = result['Percentage True'].plot.bar(color='green', rot=45)
    plt.title('Percentage of correct GPT output')
    plt.ylabel('Agreement (%)')
    plt.xlabel('Distinguishing Feature(s)')

    # Add annotations to the bars
    for idx, p in enumerate(ax.patches):
        height = p.get_height()
        total_count = result.at[result.index[idx], 'Total Counts']
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2, p.get_y() + height + 1, f'{height}%\n(n={total_count})', ha='center', va='center')

    plt.tight_layout()

    # Constructing file name from unique parts of labels
    unique_parts = set(sum((label.split(', ') for label in file_label_map.values()), []))
    feature_description = "_".join(sorted(unique_parts))
    plot_filename = os.path.join(work_dir, f'agreement_{feature_description}.png')
    plt.show()
    plt.savefig(plot_filename)
    #plt.close()
    print(f"Plot saved as: {plot_filename}")

