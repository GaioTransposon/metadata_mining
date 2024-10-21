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
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import os


# -----------------------------
# Files and Paths
# -----------------------------

home_dir = os.getenv('HOME')
work_dir = os.path.join(home_dir, "MicrobeAtlasProject")
METADATA_DIRECTORY = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"  


# Find all 'gpt_clean_output' files that end with .csv or .txt
file_patterns = ['gpt_clean_output*.csv', 'gpt_clean_output*.txt']
my_files = []
for pattern in file_patterns:
    my_files.extend(glob.glob(os.path.join(work_dir, pattern)))

print('\nNumber of files: ', len(my_files), '\n')


# Joao's file:
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/joao_biomes_parsed.csv'
joao_biomes_df = pd.read_csv(file_path, usecols=['sample', 'biome'])
joao_biomes_df['biome'] = joao_biomes_df['biome'].replace({'aquatic': 'water', 'unknown': 'other'})
joao_biomes_df['biome'].fillna('other', inplace=True)

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

def user_select_file(files):
    print("\nMultiple files found for the same label. Choose which one to keep:")
    for index, (file, _) in enumerate(files):
        print(f"{index + 1}: {file}")
    choice = int(input("Enter the number of the file to keep: ")) - 1
    return files[choice]  # Return the tuple of the chosen file and its DataFrame

# Initialize a dictionary to hold label to DataFrame mappings
label_df_map = defaultdict(list)

# Process each file, calculate agreement, and map to labels
for file_path in my_files:
    label = file_label_map[file_path]
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





################## Old (Joao's) vs new (GPT) biome agreements conf matrices: 
###
# GPT vs ground truth:
selected_biomes = ['animal', 'plant', 'soil', 'water', 'other']
lenient_agreement_df_filt = lenient_agreement_df[lenient_agreement_df['gpt_biome'].isin(selected_biomes)].copy()
lenient_agreement_df_filt.rename(columns={'gpt_biome': 'predicted_biome', 'biome': 'gd_biome'}, inplace=True)

conf_matrix_gpt = pd.crosstab(lenient_agreement_df_filt['gd_biome'], lenient_agreement_df_filt['predicted_biome'], rownames=['benchmark biome'], colnames=['predicted biome'])

###
###
# Joao's vs ground truth:  
merged_df = pd.merge(joao_biomes_df, gold_dict_df, on='sample', how='inner')
merged_df.rename(columns={'biome_x': 'predicted_biome', 'biome_y': 'gd_biome'}, inplace=True)

conf_matrix_joao = pd.crosstab(merged_df['gd_biome'], merged_df['predicted_biome'], rownames=['benchmark biome'], colnames=['predicted biome'])


# Normalize the GPT matrix
row_totals_gpt = conf_matrix_gpt.sum(axis=1)
normalized_conf_matrix_gpt = conf_matrix_gpt.div(row_totals_gpt, axis=0)

# Normalize Joao's matrix and rename 'other' to 'unknown'
row_totals_joao = conf_matrix_joao.sum(axis=1)
normalized_conf_matrix_joao = conf_matrix_joao.div(row_totals_joao, axis=0)
normalized_conf_matrix_joao.rename(columns={'other': 'unknown'}, index={'other': 'unknown'}, inplace=True)

# Define the desired order with 'unknown' for Joao and 'other' for GPT
order = ['animal', 'plant', 'soil', 'water', 'other']
order_joao = ['animal', 'plant', 'soil', 'water', 'unknown']

# Reorder the matrices
normalized_conf_matrix_gpt = normalized_conf_matrix_gpt.reindex(index=order, columns=order)
normalized_conf_matrix_joao = normalized_conf_matrix_joao.reindex(index=order_joao, columns=order_joao)

# Determine the global min and max for consistent coloring
vmin = min(normalized_conf_matrix_gpt.min().min(), normalized_conf_matrix_joao.min().min())
vmax = max(normalized_conf_matrix_gpt.max().max(), normalized_conf_matrix_joao.max().max())

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4), sharey=True)  # sharey to have same y-axis labels

# Plotting the first heatmap
sns.heatmap(normalized_conf_matrix_gpt, annot=True, fmt=".3f", cmap='viridis', ax=axes[0], vmin=vmin, vmax=vmax, cbar=False)  # No color bar here
axes[0].set_title('Benchmark biomes vs GPT biomes')
axes[0].set_xlabel('predicted biome', fontsize=11) 
axes[0].set_ylabel('benchmark biome', fontsize=11)  
axes[0].set_xticklabels(order, rotation=0, ha='center', fontsize=12)
axes[0].set_yticklabels(order, rotation=0, fontsize=12)

# Plotting the second heatmap with renamed labels
sns.heatmap(normalized_conf_matrix_joao, annot=True, fmt=".3f", cmap='viridis', ax=axes[1], vmin=vmin, vmax=vmax, cbar_ax=fig.add_axes([0.91, 0.3, 0.03, 0.4]))  # Shared color bar
axes[1].set_title('Benchmark biomes vs keyword-based classifier')
axes[1].set_xlabel('predicted biome', fontsize=11) 
axes[1].set_ylabel('benchmark biome', fontsize=11)  
axes[1].set_xticklabels(order, rotation=0, ha='center', fontsize=12)
axes[1].set_yticklabels(order, rotation=0, fontsize=12)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect to leave space for color bar
plt.show()




################## Comparison GPT matrix vs Joao matrix: 

# Chi-squared test for GPT vs. ground truth
gpt_chi2, gpt_p, _, _ = chi2_contingency(conf_matrix_gpt)
# Chi-squared test for Joao's vs. ground truth
joao_chi2, joao_p, _, _ = chi2_contingency(conf_matrix_joao)

print(f"GPT Chi-squared: {gpt_chi2}, p-value: {gpt_p}")   # low: significant difference in distribution compared to the expected frequencies.
print(f"Joao Chi-squared: {joao_chi2}, p-value: {joao_p}")   # high: greater divergence from expected frequencies

# Calculate accuracy from the confusion matrices
gpt_accuracy = (np.diag(conf_matrix_gpt).sum() / conf_matrix_gpt.values.sum()) * 100
joao_accuracy = (np.diag(conf_matrix_joao).sum() / conf_matrix_joao.values.sum()) * 100

print(f"GPT Accuracy: {gpt_accuracy:.2f}%")
print(f"João Accuracy: {joao_accuracy:.2f}%")













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

def descriptive_stats(data, label):
    print(f"\nDescriptive Statistics for {label}:")
    print(data.describe())
    print(f"\nSkewness for {label}:", data.skew())
    print(f"Kurtosis for {label}:", data.kurt())

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




################## Misclassified samples in detail 


# Display top misclassified samples 
top_misclassified = summary.sort_values(by='misclass_count', ascending=False).head(30)
print("\nTop 30 misclassified samples from the full dataset:")
print(top_misclassified)


def fetch_metadata_from_sample(sample):
    """Fetch and return metadata from a sample file based on the sample ID."""
    folder_name = f"dir_{sample[-3:]}"  # Derives folder name from the last three characters of the sample ID
    folder_path = os.path.join(METADATA_DIRECTORY, folder_name)
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as file:
        return file.read()


sample_key = input("Enter the sample key to fetch metadata and misclassification count: ")

sample_data = lenient_agreement_df[lenient_agreement_df['sample'] == sample_key]
misclassifications = (~sample_data['agreement']).sum() 
print(f"Sample '{sample_key}' was misclassified {misclassifications} times out of {len(sample_data)}.")

metadata = fetch_metadata_from_sample(sample_key)
print(f"\nMetadata for '{sample_key}':\n{metadata}")




# SRS994677              water # anaerobic sludge ; should have gone to other 105/113
# SRS2217033             animal # mock community 105/108
# SRS5304049             soil # rhizosphere 111/111
# SRS942824              soil # rhizosphere 109/110






# -----------------------------
# Quick test: Is metadata getting better with time? Based on sample ID 
# -----------------------------

# Filter to include only samples starting with ...
srs_df = lenient_agreement_df[lenient_agreement_df['sample'] .str.startswith('SRS')]

# Step 1: Extract numeric part 
srs_df['sample_id_numeric'] = srs_df['sample'].str.extract('(\d+)').astype(int)

# Step 2: Calculate quartiles and filter for the bottom 25% and top 25%sorted_df = srs_df.sort_values(by='sample_id_numeric')
first_quartile = sorted_df['sample_id_numeric'].quantile(0.25)
third_quartile = sorted_df['sample_id_numeric'].quantile(0.75)
filtered_df = sorted_df[(sorted_df['sample_id_numeric'] <= first_quartile) | (sorted_df['sample_id_numeric'] >= third_quartile)]

# Assign bins based on quartiles
filtered_df['bin'] = ['old' if x <= first_quartile else 'young' for x in filtered_df['sample_id_numeric']]

# Count the number of samples in each bin and balance the bins
bin_counts = filtered_df['bin'].value_counts()
min_count = bin_counts.min()
balanced_df = filtered_df.groupby('bin').sample(n=min_count, random_state=42)

# Step 3: Analyze the agreement rates
agreement_analysis = balanced_df.groupby('bin')['agreement'].value_counts(normalize=True).unstack().fillna(0)

print("Number of samples in each bin:")
print(balanced_df['bin'].value_counts())
print("\nAgreement analysis:")
print(agreement_analysis)


# -------------------------------
# Quick test: Is metadata getting better with time? Based on published date
# -------------------------------

def get_published_date(sample_id):
    url = f"https://www.ncbi.nlm.nih.gov/sra/?term={sample_id}"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        published_info = soup.find(text="Published")
        if published_info:
            print('looking for ', sample_id)
            return published_info.find_next().text
        return "Not found"
    except requests.RequestException:
        return "Error"

def fetch_published_dates(sample_ids):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_published_date, sample_ids))
    return dict(zip(sample_ids, results))

def extract_dates(published_dates):
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = {}
    for sample_id, value in published_dates.items():
        match = re.search(date_pattern, value)
        dates[sample_id] = match.group(0) if match else "Date not found"
    return dates



def get_consensus(agreements):
    true_count = sum(agreements)
    false_count = len(agreements) - true_count
    return True if true_count > false_count else False if false_count > true_count else None

def process_data(srs_df, extracted_dates):
    srs_df['published_date'] = srs_df['sample'].map(extracted_dates)
    srs_df = srs_df[srs_df['published_date'] != 'Date not found']
    srs_df['published_date'] = pd.to_datetime(srs_df['published_date'], errors='coerce')
    srs_df.dropna(subset=['published_date'], inplace=True)
    srs_df['bin'] = pd.qcut(srs_df['published_date'], 4, labels=['youngest', 'young', 'old', 'oldest'])
    bin_counts = srs_df['bin'].value_counts()
    balanced_df = pd.concat([srs_df[srs_df['bin'] == label].sample(n=bin_counts.min(), random_state=42) for label in bin_counts.index])
    agreement_analysis = balanced_df.pivot_table(index='bin', columns='agreement', aggfunc='size', fill_value=0)
    return balanced_df, agreement_analysis


def plot_agreement(yearly_data):
    yearly_data['Proportion True'] = yearly_data[True] / yearly_data['total']
    yearly_data['Proportion False'] = yearly_data[False] / yearly_data['total']
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_data[['Proportion False', 'Proportion True']].plot(kind='bar', stacked=True, color=['red', 'green'], ax=ax)
    ax.set_title('Yearly Proportion of Agreement (True/False)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, total in enumerate(yearly_data['total']):
        ax.text(i, 1.05, f'Total: {int(total)}', ha='center', va='bottom', fontsize=9, color='black')
    plt.show()


unique_sample_ids = srs_df['sample'].unique().tolist()
published_dates = fetch_published_dates(unique_sample_ids)
extracted_dates = extract_dates(published_dates)
balanced_df, agreement_analysis = process_data(srs_df, extracted_dates)
print("Number of samples in each bin:")
print(balanced_df['bin'].value_counts())
print("\nAgreement analysis:")
print(agreement_analysis)
plot_agreement(balanced_df)

