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



df_filtered = df[df['file_size'] <= 25000]

# Plot a histogram for the filtered data, also using 100 bins
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
plt.title('Distribution of File Sizes (up to 25,000 bytes)')
plt.xlabel('File Size (bytes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()




df_filtered = df[(df['file_size'] >= 15000) & (df['file_size'] <= 25000)]


# Plot a histogram for the filtered data, also using 100 bins
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
plt.title('Distribution of File Sizes (between 15,000 and 25000 bytes)')
plt.xlabel('File Size (bytes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


df_filtered = df[df['file_size'] >= 25000]


# Plot a histogram for the filtered data, also using 100 bins
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['file_size'], bins=100, edgecolor='black')  
plt.title('Distribution of File Sizes (from 25,000 bytes above)')
plt.xlabel('File Size (bytes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()












import os
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources (only the first time)
nltk.download('stopwords')

# Set up stopwords for English (or any other language of interest)
stop_words = set(stopwords.words('english'))

# Function to process text and count words
def get_top_words(file_path, num_words=5):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Read and convert text to lowercase
        text = re.sub(r'\W+', ' ', text)  # Replace all non-word characters with spaces
        words = text.split()  # Split into words
        words = [word for word in words if word not in stop_words]  # Filter out stopwords
        word_counts = Counter(words)  # Count word frequencies
        top_words = word_counts.most_common(num_words)  # Get the most common words
        return [word for word, count in top_words]

# Assuming df_filtered is already defined and contains the file paths and sizes
top_words_per_file = {}
common_words = set()



df_filtered = df[(df['file_size'] >= 24000)]
print(len(df_filtered))



for index, row in df_filtered.iterrows():
    file_path = row['file_path']
    top_words = get_top_words(file_path)
    top_words_per_file[file_path] = top_words
    if len(common_words) == 0:
        common_words = set(top_words)
    else:
        common_words.intersection_update(top_words)

print(f"Common words across different files: {common_words}")



# >= 24000 # 27 # unspecified
# 21000-24000 # 621 # {'unknown', 'unspecified', 'condition', 'times', 'week'}
# 20500-21000 # 1229 # {'false', 'provided', 'condition', 'true', 'week'}

# 19000-20500 # 14376
# Bin 1 (Size Range: 19000 - 19614): Most common words across different files: ['condition', 'false', 'provided', 'week', 'true']
# Bin 2 (Size Range: 19614 - 19762): Most common words across different files: ['provided', 'false', 'sample_vioscreen', 'frequency', 'diet']
# Bin 3 (Size Range: 19762 - 19872): Most common words across different files: ['provided', 'false', 'true', 'gut', 'project']
# Bin 4 (Size Range: 19872 - 19937): Most common words across different files: ['provided', 'false', 'true', 'gut', 'project']
# Bin 5 (Size Range: 19937 - 20076): Most common words across different files: ['provided', 'false', 'true', 'gut', 'project']
# Bin 6 (Size Range: 20076 - 20229): Most common words across different files: ['provided', 'false', 'condition', 'true', 'week']
# Bin 7 (Size Range: 20229 - 20309): Most common words across different files: ['provided', 'false', 'condition', 'true', 'week']
# Bin 8 (Size Range: 20309 - 20374): Most common words across different files: ['provided', 'false', 'condition', 'week', 'true']
# Bin 9 (Size Range: 20374 - 20444): Most common words across different files: ['provided', 'false', 'condition', 'week', 'true']
# Bin 10 (Size Range: 20444 - 20499): Most common words across different files: ['provided', 'false', 'condition', 'week', 'true']

# 15000-19000 # 644 ['condition', 'unknown', 'week', 'yes', 'unspecified']


df_filtered = df[(df['file_size'] >= 20500) & (df['file_size'] < 21000)]

len(df_filtered)





import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Assuming you already have df and the necessary imports
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def process_text(text):
    """Clean and process text."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    return [word for word in words if word not in stop_words]

def get_top_words(file_path, num_words=5):
    """Get the top N words from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            words = process_text(file.read())
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(num_words)]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

# Sort the DataFrame by file_size for consistent binning

df_sorted = df_filtered.sort_values(by='file_size')

# Calculate the indices for each bin to ensure each has at most 100 files
bin_size = 1500
bin_limits = range(0, len(df_sorted), bin_size)
bins = [df_sorted.iloc[bin_limits[i]:bin_limits[i+1] if i+1 < len(bin_limits) else None] for i in range(len(bin_limits))]
len(bins)

# Output the common words for each bin
for i, bin_df in enumerate(bins):
    if bin_df.empty:
        print(f"Bin {i+1} is empty. Skipping...")
        continue  # Skip processing if the bin is empty

    aggregate_words = Counter()

    for index, row in bin_df.iterrows():
        file_path = row['file_path']
        top_words = get_top_words(file_path)
        print(file_path)
        print(top_words)
        aggregate_words.update(top_words)  # Aggregate all top words from all files in the bin

    # Now determine the most common top words in the bin
    most_common_words = [word for word, _ in aggregate_words.most_common(5)]

    print(f"Bin {i+1} (Size Range: {bin_df['file_size'].min()} - {bin_df['file_size'].max()}): Most common words across different files: {most_common_words}")



