#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:11:19 2024

@author: dgaio
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/"

file_data = []

# Walk through the directories to find 'clean.txt' files and gather their path and size
for dirpath, dirnames, filenames in os.walk(base_dir):
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



# /Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/dir_493/ERS810493_clean.txt	
# file size: 15373
# according to gpt tokenizer: 
# Tokens: 8,008
# Characters: 15372


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














import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/"

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


# Sort DataFrames by file size
original_df_sorted = original_df.sort_values(by='file_size').reset_index(drop=True)
clean_df_sorted = clean_df.sort_values(by='file_size').reset_index(drop=True)

# Find the median file path
median_index_original = len(original_df_sorted) // 2
median_index_clean = len(clean_df_sorted) // 2

median_file_original = original_df_sorted.loc[median_index_original, 'file_path']
median_file_clean = clean_df_sorted.loc[median_index_clean, 'file_path']

print(f"Median file path for *.txt files: {median_file_original}")
# /Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/dir_776/SRS3616776.txt <-- 1877 chars; 564 tokens
print(f"Median file path for *clean.txt files: {median_file_clean}")
# /Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs/dir_890/SRS6434890_clean.txt <-- 1133 chars; 278 tokens

# Optional: Display total size reduction
total_size_original = original_df['file_size'].sum()
total_size_clean = clean_df['file_size'].sum()
total_reduction = total_size_original - total_size_clean
print(f"Total original size: {total_size_original} bytes")
print(f"Total clean size: {total_size_clean} bytes")
print(f"Total reduction in size: {total_size_clean/total_size_original*100} bytes")









# bash approach 

base_dir="/mnt/mnemo5/dgaio/MicrobeAtlasProject/sample.info_split_dirs/dir_*"


# median txt file (excl. clean): 
find $base_dir -type f -name "*.txt" ! -name "*clean.txt" -exec du -k {} + | sort -n -k1 > sorted_files.txt
num_files=$(cat sorted_files.txt | wc -l)
median_index=$(( (num_files + 1) / 2 ))
median_file=$(awk "NR==$median_index {print \$2}" sorted_files.txt)
echo "The file with the median size is: $median_file"
# The file with the median size is: /mnt/mnemo5/dgaio/MicrobeAtlasProject/sample.info_split_dirs/dir_607/SRS5604607.txt
# cat /mnt/mnemo5/dgaio/MicrobeAtlasProject/sample.info_split_dirs/dir_607/SRS5604607.txt
# Tokens 305 Characters 1141


# median clean file: 
find $base_dir -type f -name "*clean.txt" -exec du -k {} + | sort -n -k1 > sorted_clean_files.txt
num_files=$(cat sorted_files.txt | wc -l)
median_index=$(( (num_files + 1) / 2 ))
median_file=$(awk "NR==$median_index {print \$2}" sorted_clean_files.txt)
echo "The file with the median size is: $median_file"

