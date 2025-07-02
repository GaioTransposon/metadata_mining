#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:11:19 2024

@author: dgaio
"""


# runs as: 
# python ~/github/metadata_mining/scripts/check_metadata_sizes.py --split_dirs '~/MicrobeAtlasProject/sample_info_split_dirs/'



import os
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import numpy as np
from sklearn.utils import resample
from matplotlib.backends.backend_pdf import PdfPages
import argparse



# PART 1: Overall look at metadata files size distribution: 

parser = argparse.ArgumentParser(description='Check metadata file sizes before and after cleaning.')
parser.add_argument(
    '--split_dirs',
    default='sample_info_split_dirs_test',
    help='Directory containing split metadata files (default: sample_info_split_dirs_test)'
)

args = parser.parse_args()


base_dir = os.path.expanduser(args.split_dirs)
print(base_dir)


file_data = []

# Walk through the directories to find 'clean.txt' files and gather their path and size
for dirpath, dirnames, filenames in os.walk(base_dir):
    print(dirpath)
    if any(fname.endswith("clean.txt") for fname in filenames):
        for fname in filenames:
            if fname.endswith("clean.txt"):
                file_path = os.path.join(dirpath, fname)
                file_size = os.path.getsize(file_path)
                #if file_size > 15000:
                file_data.append({"file_path": file_path, "file_size": file_size})

# Create a DataFrame
df = pd.DataFrame(file_data)

# Sort the DataFrame by file size in descending order and keep the top 10,000 entries
df = df.sort_values(by="file_size", ascending=False) #.head(10000)



summary_stats = df.describe()
print(summary_stats)



# Boxplot distribution
plt.figure(figsize=(10, 5))
plt.boxplot(df['file_size'], vert=False)
plt.title('File sizes')
plt.xlabel('file size (bytes)')
plt.grid(True)
plt.show()


# Visualizing majority: 
df_filtered = df[df['file_size'] <= 25000] # 25000*8008/15373 --> below 13'002 tokens
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
plt.title('Distribution of File Sizes (up to 25,000 bytes)')
plt.xlabel('File Size (bytes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Optionally to inspect which files are particularly large:
df_filtered = df[(df['file_size'] >= 15000) & (df['file_size'] <= 25000)] # 15000*8008/15373 --> above 7813 tokens
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
plt.title('Distribution of File Sizes (up to 25,000 bytes)')
plt.xlabel('File Size (bytes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


df_filtered = df[df['file_size'] >= 25000]   # 25000*8008/15373 --> above 13'002 tokens
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
plt.title('Distribution of File Sizes (up to 25,000 bytes)')
plt.xlabel('File Size (bytes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()






# PART 2: Analysis of reduction based on file size (bytes) reduction: 



original_files_data = []
clean_files_data = []

# Walk through the directories to find both '*.txt' and '*clean.txt' files
for dirpath, dirnames, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith(".txt") and not filename.endswith("clean.txt"):
            file_path = os.path.join(dirpath, filename)
            file_size = os.path.getsize(file_path)
            original_files_data.append({"file_path": file_path, "file_size": file_size})
        elif filename.endswith("clean.txt"):
            file_path = os.path.join(dirpath, filename)
            file_size = os.path.getsize(file_path)
            clean_files_data.append({"file_path": file_path, "file_size": file_size})

# Create DataFrames
original_df = pd.DataFrame(original_files_data)
clean_df = pd.DataFrame(clean_files_data)

# remove log files
original_df = original_df[~original_df['file_path'].str.contains(r'/log')]
clean_df = clean_df[~clean_df['file_path'].str.contains(r'/log')]


# Sort DataFrames by file size
original_df_sorted = original_df.sort_values(by='file_size').reset_index(drop=True)
clean_df_sorted = clean_df.sort_values(by='file_size').reset_index(drop=True)

# Find the median file path
median_index_original = len(original_df_sorted) // 2
median_index_clean = len(clean_df_sorted) // 2


total_size_original = original_df['file_size'].sum()
total_size_clean = clean_df['file_size'].sum()

# Calculate the absolute reduction in bytes
total_reduction_bytes = total_size_original - total_size_clean

# Calculate the percentage reduction
total_reduction_percentage = (total_reduction_bytes / total_size_original) * 100

print(f"Total original size: {total_size_original} bytes")
print(f"Total clean size: {total_size_clean} bytes")
print(f"Total reduction in size: {total_reduction_bytes} bytes") 
print(f"Percentage reduction in size: {total_reduction_percentage:.2f}%")  # Percentage reduction in size: 34.53%




# PART 3: Analysis of reduction based on tokens reduction: 


# Initialize the tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Function to read file content and count tokens
def read_and_tokenize(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    tokens = enc.encode(content)
    return len(tokens)

    
# Function to perform bootstrap sampling from Q1 to Q3 and calculate token reduction
def bootstrap_token_reduction_iqr(original_df, clean_df, n_iterations=100, sample_size=100):
    reductions = []
    original_token_counts = []
    clean_token_counts = []
    for _ in range(n_iterations):
        # Sample with replacement from Q1 to Q3 of the original and cleaned dataframes
        original_sample = resample(original_df[(original_df['file_size'] >= original_df['file_size'].quantile(0.25)) & 
                                               (original_df['file_size'] <= original_df['file_size'].quantile(0.75))],
                                   n_samples=sample_size, replace=True)
        clean_sample = resample(clean_df[(clean_df['file_size'] >= clean_df['file_size'].quantile(0.25)) & 
                                         (clean_df['file_size'] <= clean_df['file_size'].quantile(0.75))],
                                n_samples=sample_size, replace=True)

        # Calculate the mean token counts for each sample
        original_tokens = np.array([read_and_tokenize(path) for path in original_sample['file_path']])
        clean_tokens = np.array([read_and_tokenize(path) for path in clean_sample['file_path']])
        original_mean_tokens = np.mean(original_tokens)
        clean_mean_tokens = np.mean(clean_tokens)
        
        # Store the mean token counts
        original_token_counts.append(original_mean_tokens)
        clean_token_counts.append(clean_mean_tokens)
        
        # Calculate the reduction in tokens
        token_reduction = original_mean_tokens - clean_mean_tokens
        percentage_reduction = (token_reduction / original_mean_tokens) * 100
        reductions.append(percentage_reduction)

    return reductions, np.mean(original_token_counts), np.mean(clean_token_counts)



# Perform bootstrap analysis
results, avg_original_tokens, avg_clean_tokens = bootstrap_token_reduction_iqr(original_df, clean_df)

# Calculate mean and 95% confidence interval
mean_reduction = np.mean(results)
ci_lower, ci_upper = np.percentile(results, [2.5, 97.5])

print(f"Estimated token reduction percentage due to the cleaning: {mean_reduction:.2f}%")
print(f"95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

# # run with n_iterations 100, sample_size=100
# Estimated token reduction percentage: 35.84%
# 95% Confidence Interval: [30.46%, 40.38%]
# # again: 
# Estimated token reduction percentage: 36.32%
# 95% Confidence Interval: [31.18%, 40.60%]
# # again: 
# Estimated token reduction percentage: 35.91%
# 95% Confidence Interval: [31.56%, 41.52%]


print(f"Average original tokens: {avg_original_tokens:.2f}")
print(f"Average cleaned tokens: {avg_clean_tokens:.2f}")

# 535.4 tokens * 3.8 = 2034444000 --> 2034444000*0.5/1000000 = $1017.222 input cost without cleaning
# 340.06 tokens * 3.8 = 1292228000 --> 1292228000*0.5/1000000 = $646.114 input cost after cleaning





















