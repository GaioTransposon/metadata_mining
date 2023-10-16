#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:05:41 2023

@author: dgaio
"""

import glob
import pandas as pd
import os 
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns


def interactive_file_selection(initial_pattern):
    """
    Allow user to interactively refine and select files based on their filename patterns.
    """
    current_files = glob.glob(initial_pattern)
    while True:
        # Display current files
        print("\nCurrent matching files:")
        for idx, file in enumerate(current_files, start=1):
            print(f"{idx}. {os.path.basename(file)}")

        # Ask user for further refinement or selection
        action = input("\nEnter a keyword to refine further, 'done' to finish, or 'all' to select all: ").strip().lower()
        
        if action == 'done':
            break
        elif action == 'all':
            return current_files
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
        
        # Split the content on underscores and dashes to get individual features
        labels = re.split('[_-]', content)
        
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
    return (agree_samples / total_samples) * 100


# Start with the initial pattern
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"

# 2. Open gold_dict and transform to a df
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict = gold_dict[0]
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)

initial_pattern = os.path.join(work_dir, "gpt_clean*")

selected_files = interactive_file_selection(initial_pattern)
print("\nSelected files for analysis:")
for f in selected_files:
    print(os.path.basename(f))

# Get the distinguishing features
distinguishing_features = find_distinguishing_features(selected_files)
print(distinguishing_features)



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

# Get the distinguishing features
distinguishing_features = find_distinguishing_features(selected_files)
print(distinguishing_features)

# Extract labels using the distinguishing tokens
labels = [extract_labels_from_filename(file, distinguishing_features) for file in selected_files]
print("\nLabels for the selected files:")
for file, label in zip(selected_files, labels):
    print(f"{os.path.basename(file)} -> {label}")

# [rest of the code is unchanged]



# 2. Open gold_dict and transform to a df
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict = gold_dict[0]
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['curated_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)

agreement_df = pd.DataFrame(columns=['label', 'curated_biome', 'agreement_percentage'])

# Process each file and calculate agreement percentages
for file, label in zip(selected_files, labels):
    processed_df = load_and_process_file(file, gold_dict_df)

    # Calculate agreement for each biome
    biome_agreement = processed_df.groupby('curated_biome').apply(lambda x: calculate_agreement(x)).reset_index()
    biome_agreement.columns = ['curated_biome', 'agreement_percentage']
    biome_agreement['label'] = label

    agreement_df = pd.concat([agreement_df, biome_agreement], ignore_index=True)

print(agreement_df)


# Sort the dataframe by label
agreement_df['label'] = agreement_df['label'] 
agreement_df = agreement_df.sort_values(by='label')


# Custom palette
custom_palette = {
    'animal': '#E28989',  # light_coral
    'water': '#8ADAE1',   # tiffany_blue
    'plant': '#BCD279',   # pistachio
    'soil': '#DECE8A'     # flax
}

# Plot by biome
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
bar_plot = sns.barplot(x='label', y='agreement_percentage', hue='curated_biome', data=agreement_df, 
            palette=custom_palette, ci=None, edgecolor='black', alpha=0.85)

# Annotations and labels for biome-specific plot
plt.xlabel('label', fontsize=15)
plt.ylabel('Agreement Percentage', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(title='Biome', title_fontsize='14', fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f'Agreement of GPT Predictions with Curated Biomes', fontsize=16, fontweight='bold')

# Add numbers on top of bars for biome-specific plot
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.1f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 10), 
                      textcoords = 'offset points',
                      fontsize = 12)

# Show the first plot
plt.show()

# Plot overall agreement in a new figure
plt.figure(figsize=(10, 8))

# Plot overall agreement
overall_agreement = agreement_df.groupby('label')['agreement_percentage'].mean().reset_index()

bar_plot_overall = sns.barplot(x='label', y='agreement_percentage', data=overall_agreement, 
            color='#8A9BDE', edgecolor='black', alpha=0.85)

# Annotations and labels for overall plot
plt.xlabel('label', fontsize=15)
plt.ylabel('Agreement Percentage', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(f'Overall Agreement of GPT Predictions', fontsize=16, fontweight='bold')

# Add numbers on top of bars for overall plot
for p in bar_plot_overall.patches:
    bar_plot_overall.annotate(format(p.get_height(), '.1f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 10), 
                      textcoords = 'offset points',
                      fontsize = 12)

# Show the second plot
plt.show()

