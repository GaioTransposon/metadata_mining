#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:30:18 2024

@author: dgaio
"""


import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
# def plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir):
#     # Compute full agreement statistics
#     full_true_counts = full_agreement_df.groupby('label')['agreement'].sum()
#     full_total_counts = full_agreement_df.groupby('label').size()
#     full_result = pd.DataFrame({
#         'Full True Counts': full_true_counts,
#         'Full Total Counts': full_total_counts,
#         'full match': (full_true_counts / full_total_counts * 100).round(2),
#         'full match label': (full_true_counts / full_total_counts * 100).round(2).astype(str) + '%\n(n=' + full_total_counts.astype(str) + ')'
#     })
#     print('full', full_result)
# 
#     # Compute lenient agreement statistics
#     lenient_true_counts = lenient_agreement_df.groupby('label')['agreement'].sum()
#     lenient_total_counts = lenient_agreement_df.groupby('label').size()
#     lenient_result = pd.DataFrame({
#         'Lenient True Counts': lenient_true_counts,
#         'Lenient Total Counts': lenient_total_counts,
#         'full+partial match': (lenient_true_counts / lenient_total_counts * 100).round(2),
#         'full+partial match label': (lenient_true_counts / lenient_total_counts * 100).round(2).astype(str) + '%\n(n=' + lenient_total_counts.astype(str) + ')'
#     })
#     print('full+partial', lenient_result)
# 
#     # Merge results for plotting
#     result = pd.merge(full_result, lenient_result, left_index=True, right_index=True, suffixes=('_full', '_lenient'))
# 
#     fig, ax = plt.subplots(figsize=(12, 8))
#     bar_width = 0.35
#     indices = range(len(result))
#     
#     # Plot bars and annotate with pre-formatted labels
#     for i, label in enumerate(result.index):
#         bar1 = ax.bar(i - bar_width/2, result.at[label, 'full match'], bar_width, color='green', label='Full Match' if i == 0 else "")
#         bar2 = ax.bar(i + bar_width/2, result.at[label, 'full+partial match'], bar_width, color='yellow', label='Full+Partial Match' if i == 0 else "")
#         
#         # Annotate bars with the pre-formatted labels
#         ax.text(i - bar_width/2, bar1[0].get_height() + 0.5, result.at[label, 'full match label'], ha='center')
#         ax.text(i + bar_width/2, bar2[0].get_height() + 0.5, result.at[label, 'full+partial match label'], ha='center')
#     
#     plt.title('Percentage of correct GPT output')
#     plt.ylabel('Agreement (%)')
#     plt.xlabel('Distinguishing Feature(s)')
#     plt.xticks(indices, result.index, rotation=45)
# 
#     plt.tight_layout()
# 
#     # Adjust legend position
#     plt.legend(title='', loc='right', title_fontsize='13', fontsize='11')
# 
#     # Constructing file name from unique parts of labels
#     unique_parts = set(sum((label.split(', ') for label in file_label_map.values()), []))
#     feature_description = "_".join(sorted(unique_parts))
#     plot_filename = os.path.join(work_dir, f'biome_agreement_{feature_description}.png')
#     #plt.savefig(plot_filename)
#     #plt.show()
#     print(f"Plot saved as: {plot_filename}")
#     
#     return full_result, lenient_result
# =============================================================================
    
def plot_biome_agreement(full_agreement_df, lenient_agreement_df, file_label_map, work_dir):
    # Compute full agreement statistics
    full_true_counts = full_agreement_df.groupby('label')['agreement'].sum()
    full_total_counts = full_agreement_df.groupby('label').size()
    full_mean = (full_true_counts / full_total_counts * 100).round(2)
    full_sd = full_agreement_df.groupby('label')['agreement'].std().round(2) * 100

    full_result = pd.DataFrame({
        'Full True Counts': full_true_counts,
        'Full Total Counts': full_total_counts,
        'full match': full_mean,
        'full match label': full_mean.astype(str) + '%\n(n=' + full_total_counts.astype(str) + ')',
        'mean': full_mean,
        'sd': full_sd,
        'sample_size': full_total_counts
    })
    print('full', full_result)

    # Compute lenient agreement statistics
    lenient_true_counts = lenient_agreement_df.groupby('label')['agreement'].sum()
    lenient_total_counts = lenient_agreement_df.groupby('label').size()
    lenient_mean = (lenient_true_counts / lenient_total_counts * 100).round(2)
    lenient_sd = lenient_agreement_df.groupby('label')['agreement'].std().round(2) * 100

    lenient_result = pd.DataFrame({
        'Lenient True Counts': lenient_true_counts,
        'Lenient Total Counts': lenient_total_counts,
        'full+partial match': lenient_mean,
        'full+partial match label': lenient_mean.astype(str) + '%\n(n=' + lenient_total_counts.astype(str) + ')',
        'mean': lenient_mean,
        'sd': lenient_sd,
        'sample_size': lenient_total_counts
    })
    print('full+partial', lenient_result)

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
    plt.legend(title='', loc='right', title_fontsize='13', fontsize='11')

    # Constructing file name from unique parts of labels
    unique_parts = set(sum((label.split(', ') for label in file_label_map.values()), []))
    feature_description = "_".join(sorted(unique_parts))
    plot_filename = os.path.join(work_dir, f'biome_agreement_{feature_description}.png')
    #plt.savefig(plot_filename)
    #plt.show()
    print(f"Plot saved as: {plot_filename}")

    return full_result, lenient_result



def plot_distribution_metrics(compare_results):
    euclidean_distances = [result['euclidean'] for result in compare_results.values()]
    cosine_similarities = [result['cosine'] for result in compare_results.values()]
    manhattan_distances = [result['manhattan'] for result in compare_results.values()]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.histplot(euclidean_distances, bins=30, kde=True, ax=axs[0], color='blue')
    axs[0].set_title('Euclidean distance distribution')
    sns.histplot(cosine_similarities, bins=30, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Cosine similarity distribution')
    sns.histplot(manhattan_distances, bins=30, kde=True, ax=axs[2], color='red')
    axs[2].set_title('Manhattan distance distribution')
    plt.tight_layout()
    #plt.show()
    
    
def plot_actual_vs_background(actual_similarities, background_similarities, title, avg_sim, median_sim, std_dev, MWU_stat, MWU_p_value):
    """Plots a box plot comparing actual and background cosine similarities and includes p-value."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([actual_similarities, background_similarities], notch=True, patch_artist=True, labels=['Actual', 'Background'])
    ax.set_title(title)
    ax.set_ylabel('Cosine Similarity')
    ax.text(0.95, 0.95, f'''
            avg: {avg_sim:.2f} 
            sd: {std_dev:.2f} 
            med: {median_sim:.2f} 
            MannWhitney U test\nU: {MWU_stat:.4f}
            MannWhitney U test\np-value: {MWU_p_value:.4f}
            ''', 
            horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=10)
    ax.grid(True)
    #plt.show()
    return fig


def plot_heatmap(matrix_gd, matrix_gpt, gpt_labels, gold_labels, keys_gpt_sampled, keys_gd_sampled):
    similarity_matrix = cosine_similarity(matrix_gd, matrix_gpt)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.set(font_scale=0.5)
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', ax=ax,
                          xticklabels=[gpt_labels[key] for key in keys_gpt_sampled],
                          yticklabels=[gold_labels[key] for key in keys_gd_sampled])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    plt.title('Cosine Similarity Heatmap Grouped by Category')
    plt.xlabel('Test Samples')
    plt.ylabel('Ground Truth Samples')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.20, left=0.15, right=1.00)
    #plt.show()
    return fig  


def save_figures_to_pdf(figures, base_filename, directory):
    """Saves a list of figure objects to a PDF file in the specified directory."""
    import matplotlib.backends.backend_pdf
    pdf_path = os.path.join(directory, f"{base_filename}.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()
    print(f"Saved to {pdf_path}")
    
    
    
def output_to_csv(df, filename):
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, pd.DataFrame([{}]), df])  # Jump a row between old and new data
        combined_df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)