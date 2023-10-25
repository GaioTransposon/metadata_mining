#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:05:41 2023

@author: dgaio
"""

# -----------------------------
# 1. Imports
# -----------------------------
import glob
import pandas as pd
import os
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 2. Function Definitions
# -----------------------------

def interactive_file_selection(initial_pattern):
    """
    Allow user to interactively refine and select files based on their filename patterns or indices.
    """
    current_files = glob.glob(initial_pattern)
    selected_files = []  # Initialize an empty list for selected files
    while True:
        # Display current files
        print("\nCurrent matching files:")
        for idx, file in enumerate(current_files, start=1):
            print(f"{idx}. {os.path.basename(file)}")

        # Ask user for further refinement or selection
        action = input("\nEnter a keyword to refine further (must start with string), 'done' to finish, 'all' to select all, or space-separated indices (e.g., '2 3') to select specific files: ").strip().lower()
        
        if action == 'done':
            return selected_files
        elif action == 'all':
            return current_files
        else:
            # Check if user entered indices
            indices = action.split()
            if all(idx.isdigit() for idx in indices):  # Check if all are numbers
                indices = [int(idx) - 1 for idx in indices]  # Convert to 0-based index
                selected_files = [current_files[idx] for idx in indices if 0 <= idx < len(current_files)]  # Check if index is in range
                return selected_files
            else:
                # Refine the list of files based on user input
                current_files = [f for f in current_files if action in f]
                
                if not current_files:
                    print("No files match your refined criteria. Resetting to initial files.")
                    current_files = glob.glob(initial_pattern)


def find_distinguishing_features(files):
    """
    Determine the distinguishing features between filenames.
    For each segment in the filenames, identify the segments that differ between files.
    """
    # Split the filenames into tokens based on underscores
    tokens = [os.path.basename(file).split('_')[:-2] for file in files]  # excluding the date and time segments

    # Transpose the tokens list for easy comparison
    tokens_transposed = list(zip(*tokens))

    # Find the distinguishing features
    distinguishing_tokens = set()
    for column in tokens_transposed:
        unique_tokens = set(column)
        if len(unique_tokens) > 1:  # If there's more than one unique token in this column
            distinguishing_tokens.update(unique_tokens)

    return distinguishing_tokens


def extract_labels_from_filename(filename, distinguishing_tokens):
    """
    Extract distinguishing labels from the filename.
    """
    # Extract the content between "gpt" and "_dt"
    pattern = re.compile(r'gpt(.*?)_dt')
    matches = pattern.search(filename)

    if matches:
        # Extracted content
        content = matches.group(1)
        
        # Split the content on underscores to get individual features
        labels = content.split('_')
        
        # Only keep the labels that are in distinguishing tokens
        labels = [label for label in labels if label in distinguishing_tokens]
        
        return ", ".join(labels)

    return "Unknown"

def load_and_process_file(file_name, gold_standard_df):
    dfr = pd.read_csv(file_name)
    return pd.merge(dfr, gold_standard_df, on='sample', how='inner')

def calculate_agreement(merged_df):
    total_samples = len(merged_df)
    agree_samples = len(merged_df[merged_df['gpt_generated_output_clean'] == merged_df['curated_biome']])
    agreement = (agree_samples / total_samples) * 100
    return agreement, total_samples
    total_samples = len(merged_df)
    agree_samples = len(merged_df[merged_df['gpt_generated_output_clean'] == merged_df['curated_biome']])
    agreement = (agree_samples / total_samples) * 100
    return agreement, total_samples

def custom_sort(label):
    # Use regex to split the string into its text and number components
    match = re.match(r"([a-z]+)([0-9]+\.[0-9]+)", label, re.I)
    if match:
        # If the regex finds a match, sort by the string, then by the number
        items = match.groups()
        return items[0], float(items[1])
    # If no match is found, sort only by the string
    return label, 0


# -----------------------------
# 3. Data Processing & Loading
# -----------------------------
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")

with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict = gold_dict[0]
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)


# -----------------------------
# 4. Interactive File Selection
# -----------------------------
initial_pattern = os.path.join(work_dir, "gpt_clean*")
selected_files = interactive_file_selection(initial_pattern)
print("\nSelected files for analysis:")
for f in selected_files:
    print(os.path.basename(f))
    
    
# -----------------------------
# 5. File Analysis & Data Extraction
# -----------------------------
distinguishing_features = find_distinguishing_features(selected_files)
labels = [extract_labels_from_filename(file, distinguishing_features) for file in selected_files]
labels = sorted(labels, key=custom_sort)


# Before section #6
# -----------------------------------
# Find common samples among all files
# -----------------------------------
all_samples = []
for file in selected_files:
    temp_df = pd.read_csv(file)
    all_samples.append(set(temp_df['sample']))

common_samples = set.intersection(*all_samples)


# Filter the processed DataFrames to keep only common samples
def filter_common_samples(df, common_samples):
    return df[df['sample'].isin(common_samples)]

agreement_df = pd.DataFrame(columns=['label', 'curated_biome', 'agreement_percentage'])
# Now, in your existing loop in section #6, apply this filter:
for file, label in zip(selected_files, labels):
    processed_df = load_and_process_file(file, gold_dict_df)
    processed_df = filter_common_samples(processed_df, common_samples)
    biome_agreement = processed_df.groupby('curated_biome').apply(lambda x: calculate_agreement(x)).reset_index()
    biome_agreement['agreement_percentage'] = biome_agreement[0].apply(lambda x: x[0])
    biome_agreement['total_samples'] = biome_agreement[0].apply(lambda x: x[1])
    biome_agreement.drop(columns=0, inplace=True)
    biome_agreement['label'] = label
    agreement_df = pd.concat([agreement_df, biome_agreement], ignore_index=True)


# Sort the dataframe by label
agreement_df['label'] = agreement_df['label']
agreement_df = agreement_df.sort_values(by='label')


# -----------------------------
# 7. Plotting & Visualization
# -----------------------------
custom_palette = {
    'animal': '#E28989',
    'water': '#8ADAE1',
    'plant': '#BCD279',
    'soil': '#DECE8A'
}

plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
order = labels
hue_order = list(custom_palette.keys())
bar_plot = sns.barplot(data=agreement_df, x='label', y='agreement_percentage', hue='curated_biome',
                       palette=custom_palette, ci=None, order=order, hue_order=hue_order)

# Adjust legend, title, and axis labels
bar_plot.legend(title='Biome', loc='upper left', bbox_to_anchor=(1, 1))
bar_plot.set_title('Agreement Percentage per Label per Biome')
bar_plot.set_xlabel('Label')
bar_plot.set_ylabel('Agreement Percentage')

# Annotate bars with agreement percentage and total samples
n_biomes = agreement_df['curated_biome'].nunique()
bar_width = 0.8 / n_biomes

for index, row in agreement_df.iterrows():
    # Extract relevant data
    label = row['label']
    biome = row['curated_biome']
    percentage = row['agreement_percentage']
    samples = row['total_samples']

    # Calculate the bar's x position
    label_position = order.index(label)  # Use the 'order' list for label positions
    biome_position = list(custom_palette.keys()).index(biome)
    bar_position = label_position - 0.4 + bar_width/2 + biome_position * bar_width

    # Annotation for the percentage
    bar_plot.annotate(f"{percentage:.1f}%", 
                      (bar_position, percentage + 2), 
                      ha='center', va='center',
                      fontsize=12)

    # Annotation for the sample size
    bar_plot.annotate(f"n = {samples}", 
                      (bar_position, percentage - 5), 
                      ha='center', va='center',
                      fontsize=10, color='black')


plt.tight_layout()
plt.show()




# -----------------------------
# 6. Dataframe Processing & Agreement Calculation for Overall Plot
# -----------------------------
# Creating a dataframe for the overall agreement and sample size per label
overall_agreement_df = pd.DataFrame(columns=['label', 'agreement_percentage', 'total_samples'])

for file, label in zip(selected_files, labels):
    processed_df = load_and_process_file(file, gold_dict_df)
    processed_df = filter_common_samples(processed_df, common_samples)  # Apply common samples filter
    agreement, samples = calculate_agreement(processed_df)
    overall_agreement_df = overall_agreement_df.append({
        'label': label, 
        'agreement_percentage': agreement, 
        'total_samples': samples}, 
        ignore_index=True)

# Sort the dataframe by label
overall_agreement_df['label'] = sorted(overall_agreement_df['label'], key=custom_sort)

# -----------------------------
# 7. Plotting & Visualization for Overall Plot
# -----------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

bar_plot = sns.barplot(data=overall_agreement_df, x='label', y='agreement_percentage', palette="viridis", ci=None)

# Adjust legend, title, and axis labels
bar_plot.set_title('Overall Agreement Percentage per Label')
bar_plot.set_xlabel('Label')
bar_plot.set_ylabel('Agreement Percentage')

# Annotate bars with agreement percentage and total samples
for index, bar in enumerate(bar_plot.patches):
    # Extract relevant data
    label = overall_agreement_df.iloc[index]['label']
    percentage = bar.get_height()
    samples = overall_agreement_df.iloc[index]['total_samples']

    # Annotation for the percentage
    bar_plot.annotate(f"{percentage:.1f}%", 
                      (bar.get_x() + bar.get_width() / 2, percentage + 2), 
                      ha='center', va='center',
                      fontsize=12)

    # Annotation for the sample size
    bar_plot.annotate(f"n = {samples}", 
                      (bar.get_x() + bar.get_width() / 2, percentage - 5), 
                      ha='center', va='center',
                      fontsize=10, color='black')

# Display the plot
plt.tight_layout()
plt.show()




