#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:37:48 2024

@author: dgaio
"""




# run as
# python overall_analysis.py \
#        --work_dir ~/MicrobeAtlasProject \
#        --metadata_dir sample_info_split_dirs \
#        --keyword_based_annot_file joao_biomes_parsed.csv



import os, sys, glob, pickle, re, requests
import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score,
    precision_score, recall_score, f1_score,
)

import argparse

home_dir = os.getenv('HOME')
mypath = os.path.join(home_dir, "github/metadata_mining/scripts")
sys.path.append(mypath)

from plot_biome_agreement import lenient_match
from features_process import extract_labels_from_filename, load_and_process_file, find_distinguishing_features 
import scipy.stats as stats
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# 1  Argument parsing (NEW) 
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="High-level comparison of GPT biome runs."
)
parser.add_argument(
    "--work_dir", default=".", help="Project root that contains embeddings/, gold_dict.pkl, etc."
)
parser.add_argument(
    "--metadata_dir",
    default="sample.info_split_dirs",
    help="Relative path (inside work_dir) to the directory with per-sample metadata files.",
)
parser.add_argument(
    "--keyword_based_annot_file",
    default="joao_biomes_parsed.csv",
    help="CSV (inside work_dir) with Joao’s biome assignments.",
)
args = parser.parse_args()

WORK_DIR       = os.path.abspath(args.work_dir)
METADATA_DIR   = os.path.join(WORK_DIR, args.metadata_dir)
JOAO_FILE_PATH = os.path.join(WORK_DIR, args.keyword_based_annot_file)
GOLD_DICT_PATH = os.path.join(WORK_DIR, "gold_dict.pkl")

# Add this script’s folder ( /app/scripts ) to PYTHONPATH so
#   `from plot_biome_agreement import …` continues to work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# ─────────────────────────────────────────────────────────────
# 2  Locate GPT output files
# ─────────────────────────────────────────────────────────────
file_patterns = ["gpt_clean_output*.csv", "gpt_clean_output*.txt"]
gpt_files = []
for pat in file_patterns:
    gpt_files.extend(glob.glob(os.path.join(WORK_DIR, pat)))

print(f"\nFound {len(gpt_files)} GPT output files.\n")

# ─────────────────────────────────────────────────────────────
# 3  Joao’s biome calls
# ─────────────────────────────────────────────────────────────
joao_biomes_df = pd.read_csv(
    JOAO_FILE_PATH, usecols=["sample", "biome"]
).replace({"aquatic": "water", "unknown": "other"})
joao_biomes_df["biome"].fillna("other", inplace=True)

# ─────────────────────────────────────────────────────────────
# 4  Ground-truth gold_dict
# ─────────────────────────────────────────────────────────────
with open(GOLD_DICT_PATH, "rb") as handle:
    gold_dict = pickle.load(handle)

gold_dict_df = pd.DataFrame(
    {"sample": list(gold_dict.keys()), "biome": [v[1] for v in gold_dict.values()]}
)



# -----------------------------
# Distinguishing Features
# -----------------------------

distinguishing_tokens = find_distinguishing_features(gpt_files)

# Extract and potentially edit labels
file_label_map = {file: extract_labels_from_filename(file, distinguishing_tokens) for file in gpt_files}


# -----------------------------
# Files processing and Agreement calculation
# -----------------------------

def user_select_file(files):
    print("\nMultiple files found for the same label. Choose which one to keep:")
    for index, (file, _) in enumerate(files):
        print(f"{index + 1}: {file}")
    choice = int(input("Enter the number of the file to keep: ")) - 1
    return files[choice]  # Return the tuple of the chosen file and its DataFrame

# Initialize a dictionary to hold label to DataFrame mappings
label_df_map = defaultdict(list)

# Process each file, calculate agreement, and map to labels
for file_path in gpt_files:
    label = file_label_map[file_path]
    print(file_path)
    df = load_and_process_file(file_path, gold_dict_df, label)
    df['agreement'] = df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)
    df['biome'] = df['biome'].str.strip()
    df['gpt_biome'] = df['gpt_biome'].str.strip()
    label_df_map[label].append((file_path, df))

# Select DataFrames, handling duplicates where necessary
selected_files = []
selected_dfs = []
for label, file_dfs in label_df_map.items():
    if len(file_dfs) > 1:
        chosen_file, chosen_df = user_select_file(file_dfs)
        selected_files.append(chosen_file)
        selected_dfs.append(chosen_df)
    else:
        selected_files.append(file_dfs[0][0])
        selected_dfs.append(file_dfs[0][1])

# Update my_files to reflect the actual files used in the final DataFrame
my_files = selected_files

# Concatenate all chosen DataFrames into a single DataFrame
lenient_agreement_df = pd.concat(selected_dfs, ignore_index=True)
unique_labels = lenient_agreement_df['label'].unique()

# Print results
print(f"\nIs there as many labels as there are files? {len(unique_labels) == len(my_files)}\n")




################## Overall agreement: 
overall_agreement_percentage = lenient_agreement_df['agreement'].mean() * 100

agreement_standard_deviation = lenient_agreement_df['agreement'].std() * 100

median_agreement = lenient_agreement_df['agreement'].median() * 100

n = len(lenient_agreement_df)
sem = agreement_standard_deviation / np.sqrt(n)
confidence_interval = stats.norm.interval(0.95, loc=overall_agreement_percentage, scale=sem)


cv = (agreement_standard_deviation / overall_agreement_percentage) * 100

skewness = lenient_agreement_df['agreement'].skew()
kurtosis = lenient_agreement_df['agreement'].kurt()

print(f"Overall Agreement Percentage: {overall_agreement_percentage:.2f}%")
print(f"Standard Deviation of Agreement: {agreement_standard_deviation:.2f}%")
print(f"95% Confidence Interval: {confidence_interval}")
print(f"Coefficient of Variation: {cv:.2f}%")
print(f"Median Agreement: {median_agreement:.2f}%")
print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurtosis:.2f}")




################## Old (Joao's) vs new (GPT) biome agreements conf matrices: 
###

selected_biomes = ['animal', 'plant', 'soil', 'water', 'other']

###
# GPT vs ground truth:
lenient_agreement_df_filt = lenient_agreement_df[lenient_agreement_df['gpt_biome'].isin(selected_biomes)].copy()
lenient_agreement_df_filt.rename(columns={'gpt_biome': 'predicted_biome', 'biome': 'gd_biome'}, inplace=True)

###
# Joao's vs ground truth:  
merged_df = pd.merge(joao_biomes_df, gold_dict_df, on='sample', how='inner')
merged_df = merged_df[merged_df['biome_x'].isin(selected_biomes)].copy()
merged_df.rename(columns={'biome_x': 'predicted_biome', 'biome_y': 'gd_biome'}, inplace=True)

# Find common sample IDs between GPT and João's datasets
common_samples = set(lenient_agreement_df_filt['sample']).intersection(set(merged_df['sample']))
print(f"Number of common samples between GPT and Joao: {len(common_samples)}")

# Subset both DataFrames to include only the common samples
lenient_agreement_df_common = lenient_agreement_df_filt[lenient_agreement_df_filt['sample'].isin(common_samples)]
merged_df_common = merged_df[merged_df['sample'].isin(common_samples)]


####
# Overall metrics: GPT vs Joao's  

y_true = lenient_agreement_df_common["gd_biome"]
y_pred = lenient_agreement_df_common["predicted_biome"]

print("\nGPT Model Biome-Specific Metrics:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro Precision:", precision_score(y_true, y_pred, average='macro'))
print("Macro Recall:", recall_score(y_true, y_pred, average='macro'))
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))


y_true = merged_df_common["gd_biome"]
y_pred = merged_df_common["predicted_biome"]
print("\nJoão's Model Biome-Specific Metrics:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro Precision:", precision_score(y_true, y_pred, average='macro'))
print("Macro Recall:", recall_score(y_true, y_pred, average='macro'))
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))
####




### Min and Max of metrics across tests 
grouped_metrics = []

for label, group in lenient_agreement_df_common.groupby("label"):
    y_true = group["gd_biome"]
    y_pred = group["predicted_biome"]

    metrics = {
        "label": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    grouped_metrics.append(metrics)

metrics_df = pd.DataFrame(grouped_metrics)

print(f"\nNumber of tests (labels): {len(metrics_df)}")

print("\nRanges of metrics per label:")
for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
    print(f"{metric}: min = {metrics_df[metric].min():.3f}, max = {metrics_df[metric].max():.3f}")
###




### biome-specific metrics:
    
    
###

# Create a confusion matrix for GPT predictions on the common samples
conf_matrix_gpt_common = pd.crosstab(lenient_agreement_df_common['gd_biome'], lenient_agreement_df_common['predicted_biome'], rownames=['benchmark biome'], colnames=['predicted biome'])
normalized_conf_matrix_gpt_common = conf_matrix_gpt_common.div(
    conf_matrix_gpt_common.sum(axis=1), axis=0
).reindex(index=selected_biomes, columns=selected_biomes).fillna(0)

# Create a confusion matrix for João's predictions on the common samples
conf_matrix_joao_common = pd.crosstab(merged_df_common['gd_biome'], merged_df_common['predicted_biome'], rownames=['benchmark biome'], colnames=['predicted biome'])
normalized_conf_matrix_joao_common = conf_matrix_joao_common.div(
    conf_matrix_joao_common.sum(axis=1), axis=0
).reindex(index=selected_biomes, columns=selected_biomes).fillna(0)

# Print matrices for inspection (optional)
print("\nNormalized GPT Confusion Matrix:\n", normalized_conf_matrix_gpt_common.round(3))
print("\nNormalized Joao Confusion Matrix:\n", normalized_conf_matrix_joao_common.round(3))

###

def compute_scores(conf_matrix):
    true_positives = conf_matrix.values.diagonal()
    total_predicted = conf_matrix.sum(axis=0).values
    total_actual = conf_matrix.sum(axis=1).values  # for per-biome accuracy

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(total_predicted != 0, true_positives / total_predicted, 0)
        per_biome_accuracy = np.where(total_actual != 0, true_positives / total_actual, 0)
        f1_scores = np.where(
            (precision + per_biome_accuracy) != 0,
            2 * (precision * per_biome_accuracy) / (precision + per_biome_accuracy),
            0
        )

    return precision, per_biome_accuracy, f1_scores


# Reindex conf_matrix_* before computing scores
conf_matrix_gpt_common = conf_matrix_gpt_common.reindex(index=selected_biomes, columns=selected_biomes).fillna(0)
conf_matrix_joao_common = conf_matrix_joao_common.reindex(index=selected_biomes, columns=selected_biomes).fillna(0)

# Compute scores
precision_gpt, accuracy_gpt, f1_scores_gpt = compute_scores(conf_matrix_gpt_common)
precision_joao, accuracy_joao, f1_scores_joao = compute_scores(conf_matrix_joao_common)

# Print per-biome metrics
print("\nPer-Biome Metrics for GPT:")
for biome, prec, acc, f1 in zip(selected_biomes, precision_gpt, accuracy_gpt, f1_scores_gpt):
    print(f"{biome.capitalize():<10}  Precision: {prec:.3f}  Accuracy: {acc:.3f}  F1: {f1:.3f}")

print("\nPer-Biome Metrics for João:")
for biome, prec, acc, f1 in zip(selected_biomes, precision_joao, accuracy_joao, f1_scores_joao):
    print(f"{biome.capitalize():<10}  Precision: {prec:.3f}  Accuracy: {acc:.3f}  F1: {f1:.3f}")


    

## Cohen's kappa: 

def extract_labels_from_conf_matrix(conf_matrix):
    actual = []
    predicted = []
    for index, row in conf_matrix.iterrows():
        for col, count in row.iteritems():
            actual.extend([index] * count)
            predicted.extend([col] * count)
    return actual, predicted

gpt_actual, gpt_predicted = extract_labels_from_conf_matrix(conf_matrix_gpt_common)
joao_actual, joao_predicted = extract_labels_from_conf_matrix(conf_matrix_joao_common)

gpt_kappa = cohen_kappa_score(gpt_actual, gpt_predicted)
joao_kappa = cohen_kappa_score(joao_actual, joao_predicted)

print(f"GPT Kappa: {gpt_kappa:.3f}")   # -1 to 1: 1 is perfect agreement
print(f"João Kappa: {joao_kappa:.3f}")
## 





###
# Plotting confusion matrices: 

biome_order = ["animal", "plant", "soil", "water", "other"]    

# Define font properties for heatmap annotations
annot_font = {
    'size': 10,
    'weight': 'normal',
    'color': 'white',
    'family': 'times'
}

# Create the figure with GridSpec for better control, adjusting for A4 width and spacing
fig = plt.figure(figsize=(13, 5))  # Aiming for A4 width
gs = gridspec.GridSpec(2, 7, height_ratios=[4, 0.5], width_ratios=[3, 3, 0.25, 0.5, 1.3, 1.3, 0.4], hspace=0.3)

# Heatmap for GPT predictions
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(normalized_conf_matrix_gpt_common, annot=True, fmt=".3f", cmap='viridis', ax=ax1, cbar=False, 
            vmin=0, vmax=1, annot_kws=annot_font)
ax1.set_title('GPT\nclassification accuracy', fontsize=10)
ax1.set_xlabel('Predicted biome', fontsize=10)
ax1.set_ylabel('Benchmark biome', fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=10)

# Heatmap for João's predictions
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(normalized_conf_matrix_joao_common, annot=True, fmt=".3f", cmap='viridis', ax=ax2, cbar=False, 
            vmin=0, vmax=1, annot_kws=annot_font)
ax2.set_title('Keyword-based classifier\nclassification accuracy', fontsize=10)
ax2.set_xlabel('Predicted biome', fontsize=10)
ax2.set_ylabel(' \n \n  ')
ax2.set_yticklabels([])  # Hide y-axis labels to avoid repetition
ax2.tick_params(axis='x', which='major', labelsize=10)

# Color bar in its own dedicated GridSpec cell with more width
cbar_ax = fig.add_subplot(gs[0, 2])
cbar = fig.colorbar(ax1.collections[0], cax=cbar_ax, orientation='vertical')
cbar.set_label(' ', fontsize=9)  # Set the label font size
cbar.ax.tick_params(labelsize=9)  # Set the tick label font size

# Spacer (empty) column to create additional space between color bar and bar plots
spacer_ax = fig.add_subplot(gs[0, 3])
spacer_ax.axis("off")  # Hide this axis

# Precision bar plot with increased spacing on the right
ax3 = fig.add_subplot(gs[0, 4])
bar_width = 0.35
y = np.arange(len(biome_order))  # Label locations
ax3.barh(y - bar_width/2, precision_gpt, height=bar_width, label='GPT', color='#b204f8')
ax3.barh(y + bar_width/2, precision_joao, height=bar_width, label='Keyword-based', color='#fab406')
ax3.set_title('Precision', fontsize=10)
ax3.set_yticks(y)
ax3.set_yticklabels(biome_order)
ax3.invert_yaxis()  # Keep y-axis inverted to match the layout
ax3.tick_params(axis='both', which='major', labelsize=9)

# F1 Score bar plot
ax4 = fig.add_subplot(gs[0, 5])
ax4.barh(y - bar_width/2, f1_scores_gpt, height=bar_width, label='GPT', color='#b204f8')
ax4.barh(y + bar_width/2, f1_scores_joao, height=bar_width, label='Keyword-based\nclassifier', color='#fab406')
ax4.set_title('F1 Score', fontsize=10)
ax4.set_yticks(y)
ax4.set_yticklabels([])  # Hide y-axis labels to save space
ax4.invert_yaxis()  # Maintain alignment with other bar plot
ax4.tick_params(axis='both', which='major', labelsize=9)

# Add the legend in the bottom row, centered under the bar plots
legend_ax = fig.add_subplot(gs[1, 4:6])  # Span across columns for better centering
legend = legend_ax.legend(*ax4.get_legend_handles_labels(), loc='center', fontsize=9)
legend_ax.axis('off')  # Hide the axis box for the legend

plt.show()
###




################################################################################
################################################################################
################## Are some biomes harder to predict than others? 
agreement_by_biome = lenient_agreement_df.groupby('biome')['agreement'].mean().sort_values()*100

print('\nAgreement by biome: ', agreement_by_biome, '\n')

plt.figure(figsize=(10, 6))
ax = agreement_by_biome.plot(kind='bar', color='skyblue')
plt.title('Agreement by biome')
plt.xlabel('biome')
plt.ylabel('agreement (%)')
plt.ylim(0, 100)
# add text labels on bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


# We are interested about False agreement samples, so filter to keep: 
false_agreements = lenient_agreement_df[lenient_agreement_df['agreement'] == False]



################## And where is this biome-bias? Confusion matrices: 

########## 1. Overall confusion matrix
conf_matrix_false = pd.crosstab(false_agreements['biome'], false_agreements['gpt_biome'], rownames=['Actual Biome'], colnames=['Predicted Biome'])

# Visualize the confusion matrix using a heatmap without annotations
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_false, cmap='Blues')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
plt.yticks(rotation=0)  # Ensure y labels are horizontal
plt.title('Overall confusion matrix (False agreements only)')
plt.show()
##

########## 2. Filtered confusion matrix:
# remove if seen less than twice (irrelevant to this purpose)
mask = conf_matrix_false <= 2
filtered_conf_matrix = conf_matrix_false.mask(mask)

# drop empty 
filtered_conf_matrix = filtered_conf_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

# replace NaNs with 0 (for display purposes)
filtered_conf_matrix = filtered_conf_matrix.fillna(0).astype(int)

# plt.figure(figsize=(12, 10))
# sns.heatmap(filtered_conf_matrix, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 10},
#             linewidths=.5, cbar=False)  # Hide the color bar if not needed
# plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
# plt.yticks(rotation=0)  # Keep y labels horizontal
# plt.title('Filtered Confusion Matrix of Biome Predictions (Counts > 2)')
# plt.show()
# ##

########## 3. Filtered and normalized confusion matrix:
# normalize by row (actual Biome)
row_totals = filtered_conf_matrix.sum(axis=1)
normalized_conf_matrix = filtered_conf_matrix.div(row_totals, axis=0)

plt.figure(figsize=(12, 10))
sns.heatmap(normalized_conf_matrix, annot=True, fmt=".3f", cmap='viridis')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Filtered Confusion Matrix of Biome Predictions (Counts > 2) - Normalized by Row')
plt.show()
##

##################




################## Are certain biomes consistently misclassified more often than what would occur by random chance? ##################

# Data loading and initial processing
contingency_table = pd.crosstab(false_agreements['biome'], false_agreements['gpt_biome'])
expected = chi2_contingency(contingency_table)[3]
expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

# Helper functions for repeated calculations
def calculate_residuals(contingency, expected):
    residuals = contingency - expected
    return residuals / np.sqrt(expected)


# Analysis of Biomes
print(f"Chi-squared: {chi2_contingency(contingency_table)[:2]}")
standardized_residuals = calculate_residuals(contingency_table, expected_df)
misclassification_percentages = (contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100)

print("\nPercentages of Misclassifications (Top 6 Percentages per Biome):")
for biome in misclassification_percentages.index:
    top_percentages = misclassification_percentages.loc[biome].nlargest(6)
    if not top_percentages.empty:
        print(f"\n{biome}:")
        for gpt_biome, percentage in top_percentages.items():
            residual = standardized_residuals.at[biome, gpt_biome]
            print(f"{gpt_biome:10} {percentage:.2f}% residual {residual:.2f}")



def descriptive_stats(data, label):
    print(f"\nDescriptive Statistics for {label}:")
    print(data.describe())
    print(f"\nSkewness for {label}:", data.skew())
    print(f"Kurtosis for {label}:", data.kurt())


# Misclassified samples analysis
misclass_counts = lenient_agreement_df[lenient_agreement_df['agreement'] == False].groupby('sample')['agreement'].count()
common_misclassifications = lenient_agreement_df.groupby('sample')['gpt_biome'].agg(pd.Series.mode)
summary = pd.DataFrame({
    'misclass_count': misclass_counts,
    'most_common_misclass': common_misclassifications
})


plt.figure(figsize=(10, 6))
plt.hist(summary['misclass_count'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Misclassifications per Sample (All Data)')
plt.xlabel('Misclassifications Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


descriptive_stats(misclass_counts, 'All Misclassifications')


# Compute IQR
Q1 = misclass_counts.quantile(0.25)
Q3 = misclass_counts.quantile(0.75)
IQR = Q3 - Q1

# Define outliers as values greater than Q3 + 1.5*IQR
outlier_threshold = Q3 + 1.5 * IQR
outliers = misclass_counts[misclass_counts > outlier_threshold]

print(f"\nOutlier threshold: {outlier_threshold}")
print(f"Number of outlier samples: {len(outliers)}")
print(f"Proportion of outlier samples: {len(outliers) / len(misclass_counts):.2%}")


percentile_95 = misclass_counts.quantile(0.95)
top_5_percent = misclass_counts[misclass_counts >= percentile_95]

print(f"\n95th percentile threshold: {percentile_95}")
print(f"Number of samples above 95th percentile: {len(top_5_percent)}")


# =============================================================================
# ################## Study misclassified samples in detail 
# 
# 
# # Display top misclassified samples 
# top_misclassified = summary.sort_values(by='misclass_count', ascending=False).head(50)
# print("\nTop 50 misclassified samples from the full dataset:")
# print(top_misclassified)
# 
# 
# def fetch_metadata_from_sample(sample):
#     """Fetch and return metadata from a sample file based on the sample ID."""
#     folder_name = f"dir_{sample[-3:]}"  # Derives folder name from the last three characters of the sample ID
#     folder_path = os.path.join(METADATA_DIRECTORY, folder_name)
#     metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
#     with open(metadata_file_path, 'r') as file:
#         return file.read()
# 
# 
# sample_key = input("Enter the sample key to fetch metadata and misclassification count: ")
# 
# sample_data = lenient_agreement_df[lenient_agreement_df['sample'] == sample_key]
# misclassifications = (~sample_data['agreement']).sum() 
# print(f"Sample '{sample_key}' was misclassified {misclassifications} times out of {len(sample_data)}.")
# 
# metadata = fetch_metadata_from_sample(sample_key)
# print(f"\nMetadata for '{sample_key}':\n{metadata}")
# 
# 
# 
# 
# # SRS994677              water # anaerobic sludge ; should have gone to other 105/113
# # SRS2217033             animal # mock community 105/108
# # SRS4776621             soil # rhizosphere 94/95
# # SRS942824              soil # rhizosphere 109/110
# 
# =============================================================================




# =============================================================================
# # -----------------------------
# # Quick test: Is metadata getting better with time? Based on sample ID 
# # -----------------------------
# 
# # Filter to include only samples starting with ...
# srs_df = lenient_agreement_df[lenient_agreement_df['sample'] .str.startswith('SRS')]
# 
# # Step 1: Extract numeric part 
# srs_df['sample_id_numeric'] = srs_df['sample'].str.extract('(\d+)').astype(int)
# 
# # Step 2: Calculate quartiles and filter for the bottom 25% and top 25%sorted_df = srs_df.sort_values(by='sample_id_numeric')
# first_quartile = sorted_df['sample_id_numeric'].quantile(0.25)
# third_quartile = sorted_df['sample_id_numeric'].quantile(0.75)
# filtered_df = sorted_df[(sorted_df['sample_id_numeric'] <= first_quartile) | (sorted_df['sample_id_numeric'] >= third_quartile)]
# 
# # Assign bins based on quartiles
# filtered_df['bin'] = ['old' if x <= first_quartile else 'young' for x in filtered_df['sample_id_numeric']]
# 
# # Count the number of samples in each bin and balance the bins
# bin_counts = filtered_df['bin'].value_counts()
# min_count = bin_counts.min()
# balanced_df = filtered_df.groupby('bin').sample(n=min_count, random_state=42)
# 
# # Step 3: Analyze the agreement rates
# agreement_analysis = balanced_df.groupby('bin')['agreement'].value_counts(normalize=True).unstack().fillna(0)
# 
# print("Number of samples in each bin:")
# print(balanced_df['bin'].value_counts())
# print("\nAgreement analysis:")
# print(agreement_analysis)
# 
# 
# # -------------------------------
# # Quick test: Is metadata getting better with time? Based on published date
# # -------------------------------
# 
# def get_published_date(sample_id):
#     url = f"https://www.ncbi.nlm.nih.gov/sra/?term={sample_id}"
#     try:
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         published_info = soup.find(text="Published")
#         if published_info:
#             print('looking for ', sample_id)
#             return published_info.find_next().text
#         return "Not found"
#     except requests.RequestException:
#         return "Error"
# 
# def fetch_published_dates(sample_ids):
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         results = list(executor.map(get_published_date, sample_ids))
#     return dict(zip(sample_ids, results))
# 
# def extract_dates(published_dates):
#     date_pattern = r'\d{4}-\d{2}-\d{2}'
#     dates = {}
#     for sample_id, value in published_dates.items():
#         match = re.search(date_pattern, value)
#         dates[sample_id] = match.group(0) if match else "Date not found"
#     return dates
# 
# 
# 
# def get_consensus(agreements):
#     true_count = sum(agreements)
#     false_count = len(agreements) - true_count
#     return True if true_count > false_count else False if false_count > true_count else None
# 
# def process_data(srs_df, extracted_dates):
#     srs_df['published_date'] = srs_df['sample'].map(extracted_dates)
#     srs_df = srs_df[srs_df['published_date'] != 'Date not found']
#     srs_df['published_date'] = pd.to_datetime(srs_df['published_date'], errors='coerce')
#     srs_df.dropna(subset=['published_date'], inplace=True)
#     srs_df['bin'] = pd.qcut(srs_df['published_date'], 4, labels=['youngest', 'young', 'old', 'oldest'])
#     bin_counts = srs_df['bin'].value_counts()
#     balanced_df = pd.concat([srs_df[srs_df['bin'] == label].sample(n=bin_counts.min(), random_state=42) for label in bin_counts.index])
#     agreement_analysis = balanced_df.pivot_table(index='bin', columns='agreement', aggfunc='size', fill_value=0)
#     return balanced_df, agreement_analysis
# 
# 
# def plot_agreement(yearly_data):
#     yearly_data['Proportion True'] = yearly_data[True] / yearly_data['total']
#     yearly_data['Proportion False'] = yearly_data[False] / yearly_data['total']
#     fig, ax = plt.subplots(figsize=(10, 6))
#     yearly_data[['Proportion False', 'Proportion True']].plot(kind='bar', stacked=True, color=['red', 'green'], ax=ax)
#     ax.set_title('Yearly Proportion of Agreement (True/False)')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Proportion')
#     plt.xticks(rotation=45)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     for i, total in enumerate(yearly_data['total']):
#         ax.text(i, 1.05, f'Total: {int(total)}', ha='center', va='bottom', fontsize=9, color='black')
#     plt.show()
# 
# 
# unique_sample_ids = srs_df['sample'].unique().tolist()
# published_dates = fetch_published_dates(unique_sample_ids)
# extracted_dates = extract_dates(published_dates)
# balanced_df, agreement_analysis = process_data(srs_df, extracted_dates)
# print("Number of samples in each bin:")
# print(balanced_df['bin'].value_counts())
# print("\nAgreement analysis:")
# print(agreement_analysis)
# plot_agreement(balanced_df)
# =============================================================================

