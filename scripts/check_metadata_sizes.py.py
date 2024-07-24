#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:11:19 2024

@author: dgaio
"""
# =============================================================================
# 
# 
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# base_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/"
# 
# file_data = []
# 
# # Walk through the directories to find 'clean.txt' files and gather their path and size
# for dirpath, dirnames, filenames in os.walk(base_dir):
#     if any(fname.endswith("clean.txt") for fname in filenames):
#         for fname in filenames:
#             if fname.endswith("clean.txt"):
#                 file_path = os.path.join(dirpath, fname)
#                 file_size = os.path.getsize(file_path)
#                 #if file_size > 15000:
#                 file_data.append({"file_path": file_path, "file_size": file_size})
# 
# # Create a DataFrame
# df = pd.DataFrame(file_data)
# 
# # Sort the DataFrame by file size in descending order and keep the top 10,000 entries
# df = df.sort_values(by="file_size", ascending=False) #.head(10000)
# 
# 
# 
# summary_stats = df.describe()
# print(summary_stats)
# 
# # Boxplot distribution
# plt.figure(figsize=(10, 5))
# plt.boxplot(df['file_size'], vert=False)
# plt.title('File sizes')
# plt.xlabel('file size (bytes)')
# plt.grid(True)
# plt.show()
# 
# 
# 
# # /Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/dir_493/ERS810493_clean.txt	
# # file size: 15373
# # according to gpt tokenizer: 
# # Tokens: 8,008
# # Characters: 15372
# 
# 
# # Visualizing majority: 
# df_filtered = df[df['file_size'] <= 25000] # 25000*8008/15373 --> below 13'002 tokens
# plt.figure(figsize=(10, 6))
# plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
# plt.title('Distribution of File Sizes (up to 25,000 bytes)')
# plt.xlabel('File Size (bytes)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# 
# 
# # Optionally to inspect which files are particularly large:
# df_filtered = df[(df['file_size'] >= 15000) & (df['file_size'] <= 25000)] # 15000*8008/15373 --> above 7813 tokens
# plt.figure(figsize=(10, 6))
# plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
# plt.title('Distribution of File Sizes (up to 25,000 bytes)')
# plt.xlabel('File Size (bytes)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# 
# 
# df_filtered = df[df['file_size'] >= 25000]   # 25000*8008/15373 --> above 13'002 tokens
# plt.figure(figsize=(10, 6))
# plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
# plt.title('Distribution of File Sizes (up to 25,000 bytes)')
# plt.xlabel('File Size (bytes)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# 
# 
# 
# 
# =============================================================================









import os
import glob

def calculate_total_size(path_pattern):
    total_size = 0
    # Find all files matching the path pattern
    for filename in glob.glob(path_pattern, recursive=True):
        # Get the size of each file and add it to the total
        total_size += os.path.getsize(filename)
    return total_size

# Define the base directory and file patterns
#base_directory = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/"
base_directory = "/Users/dgaio/MicrobeAtlasProject/sample.info_split_dirs/"

txt_pattern = base_directory + 'dir_*/**/*.txt'
clean_txt_pattern = base_directory + 'dir_*/**/*clean.txt'

# Calculate the total sizes
total_size_txt = calculate_total_size(txt_pattern)
total_size_clean_txt = calculate_total_size(clean_txt_pattern)

print(f"Total size of *.txt files: {total_size_txt} bytes")
print(f"Total size of *clean.txt files: {total_size_clean_txt} bytes")









