#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:37:48 2024

@author: dgaio
"""


import sys
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from plot_biome_agreement import lenient_match
from features_process import extract_labels_from_filename, load_and_process_file, find_distinguishing_features 
import numpy as np
import scipy.stats as stats


# -----------------------------
# Files and Paths
# -----------------------------

home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")

# Find all 'gpt_clean_output' files that end with .csv or .txt
file_patterns = ['gpt_clean_output*.csv', 'gpt_clean_output*.txt']
my_files = []
for pattern in file_patterns:
    my_files.extend(glob.glob(os.path.join(work_dir, pattern)))

print('\nNumber of files: ', len(my_files), '\n')


# -----------------------------
# Ground truth loading & processing
# -----------------------------

input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict_df = pd.DataFrame({
    'sample': [k for k, v in gold_dict.items()],
    'biome': [v[1] for k, v in gold_dict.items()]})


# -----------------------------
# Distinguishing Features
# -----------------------------

distinguishing_tokens = find_distinguishing_features(my_files)

# Extract and potentially edit labels
file_label_map = {file: extract_labels_from_filename(file, distinguishing_tokens) for file in my_files}


# -----------------------------
# Files processing and Agreement calculation
# -----------------------------

all_dfs = []
for file_path in my_files:
    labels = file_label_map[file_path]
    df = load_and_process_file(file_path, gold_dict_df, labels)
    df['agreement'] = df.apply(lambda row: lenient_match(row['biome'], row['gpt_biome']), axis=1)
    all_dfs.append(df)


lenient_agreement_df = pd.concat(all_dfs, ignore_index=True)

# Strip whitespace from biome labels in the DataFrame
lenient_agreement_df['biome'] = lenient_agreement_df['biome'].str.strip()
lenient_agreement_df['gpt_biome'] = lenient_agreement_df['gpt_biome'].str.strip()


unique_labels = lenient_agreement_df['label'].unique()
print('\nIs there as many labels as there are files? ', len(unique_labels) == len(my_files), '\n')






################## Overall agreement: 
overall_agreement_percentage = lenient_agreement_df['agreement'].mean() * 100

agreement_standard_deviation = lenient_agreement_df['agreement'].std() * 100

median_agreement = lenient_agreement_df['agreement'].median() * 100

confidence_interval = stats.norm.interval(0.95, loc=overall_agreement_percentage, scale=agreement_standard_deviation)

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











################## Are some biomes harder to predict than others? 
agreement_by_biome = lenient_agreement_df.groupby('biome')['agreement'].mean().sort_values()

print('\nAgreement by biome: ', agreement_by_biome, '\n')

plt.figure(figsize=(10, 6))
ax = agreement_by_biome.plot(kind='bar', color='skyblue')
plt.title('Agreement by biome')
plt.xlabel('biome')
plt.ylabel('agreement (%)')
plt.ylim(0, 1)
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

# # Visualize the confusion matrix using a heatmap without annotations
# plt.figure(figsize=(12, 10))
# sns.heatmap(conf_matrix_false, cmap='Blues')
# plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
# plt.yticks(rotation=0)  # Ensure y labels are horizontal
# plt.title('Overall confusion matrix (False agreements only)')
# plt.show()
# ##

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

# Create a contingency table from the false agreements data
contingency_table = pd.crosstab(false_agreements['biome'], false_agreements['gpt_biome'])

# Perform the Chi-squared test for independence
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared: {chi2}, p-value: {p_value}")

# Convert the expected array to a DataFrame with similar indexing as the contingency_table
expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

# Calculate residuals (Observed - Expected)
residuals = contingency_table - expected_df

# Calculate standardized residuals ((Observed - Expected) / sqrt(Expected))
standardized_residuals = residuals / np.sqrt(expected_df)

# More insights into residuals: 

# Identify and isolate significant residuals
significant_residuals = standardized_residuals[(standardized_residuals > 5) | (standardized_residuals < -5)]

# Highlight the most extreme cases
sorted_significant_residuals = significant_residuals.unstack().dropna().abs().sort_values(ascending=False)

for index, value in sorted_significant_residuals.iteritems():
    actual_biome, predicted_biome = index
    try:
        original_value = standardized_residuals.at[actual_biome, predicted_biome]
        print(f"Actual Biome: {actual_biome}, Predicted Biome: {predicted_biome}, Residual: {original_value:.2f}")
    except KeyError:
        print(f"KeyError encountered for Biome: {actual_biome} or {predicted_biome}")








################## Misclassified samples: 

# Count misclassifications per sample
misclass_counts = false_agreements.groupby('sample')['agreement'].count()

# Find the most common incorrect biome predicted for each sample
common_misclassifications = false_agreements.groupby('sample')['gpt_biome'].agg(pd.Series.mode)

# Combine these into a single DataFrame for analysis
misclassification_summary = pd.DataFrame({
    'misclass_count': misclass_counts,
    'most_common_misclass': common_misclassifications
})

# Samples vcs frequency of misclassifications
top_misclassified_samples = misclassification_summary.sort_values(by='misclass_count', ascending=False)
# ax = top_misclassified_samples['misclass_count'].plot(kind='bar', color='red', figsize=(10, 6))
# plt.title('misclassified samples')
# plt.xlabel('sample id')
# plt.ylabel('Misclassifications counts')
# ax.set_xticklabels([]) 
# plt.show()

top_20_misclassified = top_misclassified_samples.head(20)
print("Top 20 misclassified samples:")
print(top_20_misclassified)

plt.figure(figsize=(10, 6))
plt.hist(misclassification_summary['misclass_count'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of misclassifications per sample')
plt.xlabel('misclassifications count')
plt.ylabel('frequency')
plt.grid(True)
plt.show()

# Stats of misclassifications
print("\nDescriptive Statistics:")
print(misclass_counts.describe())
print("\nSkewness:", misclass_counts.skew())
print("Kurtosis:", misclass_counts.kurt())




################## Misclassified samples: 

# Count misclassifications per sample
misclass_counts = false_agreements.groupby('sample')['agreement'].count()

# Filter out samples misclassified only once
filtered_misclass_counts = misclass_counts[misclass_counts > 1]

# Find the most common incorrect biome predicted for each sample
common_misclassifications = false_agreements[false_agreements['sample'].isin(filtered_misclass_counts.index)].groupby('sample')['gpt_biome'].agg(pd.Series.mode)

# Combine these into a single DataFrame for analysis
misclassification_summary = pd.DataFrame({
    'misclass_count': filtered_misclass_counts,
    'most_common_misclass': common_misclassifications
})

# Samples vs frequency of misclassifications
top_misclassified_samples = misclassification_summary.sort_values(by='misclass_count', ascending=False)
# ax = top_misclassified_samples['misclass_count'].plot(kind='bar', color='red', figsize=(10, 6))
# plt.title('misclassified samples')
# plt.xlabel('sample id')
# plt.ylabel('Misclassifications counts')
# ax.set_xticklabels([]) 
# plt.show()

top_20_misclassified = top_misclassified_samples.head(20)
print("Top 20 misclassified samples:")
print(top_20_misclassified)

plt.figure(figsize=(10, 6))
plt.hist(misclassification_summary['misclass_count'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of misclassifications per sample')
plt.xlabel('misclassifications count')
plt.ylabel('frequency')
plt.grid(True)
plt.show()

# Stats of misclassifications
print("\nDescriptive Statistics:")
print(filtered_misclass_counts.describe())
print("\nSkewness:", filtered_misclass_counts.skew())
print("Kurtosis:", filtered_misclass_counts.kurt())


