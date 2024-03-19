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
from sklearn.metrics import confusion_matrix
import numpy as np
import textwrap
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



# =============================================================================
# def load_and_process_file(file_name, gold_standard_df):
#     # Load the file without headers, since the number of columns can vary
#     dfr = pd.read_csv(file_name, header=None)
# 
#     # Select only the first and second columns for processing
#     # Assuming the first column is 'sample' and the second is 'gpt_generated_output_clean'
#     dfr = dfr.iloc[:, [0, 1]]
#     dfr.columns = ['sample', 'gpt_generated_output_clean']  # Assign column names after selection
# 
#     # Handle missing values in 'gpt_generated_output_clean'
#     dfr['gpt_generated_output_clean'] = dfr['gpt_generated_output_clean'].fillna('unknown').astype(str)
# 
#     # Merge with the gold standard DataFrame
#     merged_df = pd.merge(dfr, gold_standard_df, on='sample', how='inner')
# 
#     # Check for 'unknown' in the merged DataFrame
#     num_unknown_merged = merged_df['gpt_generated_output_clean'].value_counts().get('unknown', 0)
#     print(f"Number of 'unknown' after merging '{os.path.basename(file_name)}':", num_unknown_merged)
# 
#     return merged_df
# =============================================================================

def load_and_process_file(file_name, gold_standard_df, label):
    # Load the file without headers, as the number of columns can vary
    dfr = pd.read_csv(file_name, header=None)

    # Select only the first and second columns for processing
    dfr = dfr.iloc[:, [0, 1]]
    dfr.columns = ['sample', 'gpt_biome']  # Rename columns for clarity

    # Add the label column
    dfr['label'] = label

    # Merge with the gold standard DataFrame
    merged_df = pd.merge(dfr, gold_standard_df, on='sample', how='inner')

    return merged_df


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
gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)
#gold_dict_df = gold_dict_df[gold_dict_df['biome'] != 'unknown']




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
print('Distinguishing features of GPT files are: ', labels)




##########

# plotting agreement 

# Concatenate all DataFrames
all_dfs = []
for file, label in zip(selected_files, labels):
    processed_df = load_and_process_file(file, gold_dict_df, label)
    all_dfs.append(processed_df)

concatenated_df = pd.concat(all_dfs, ignore_index=True)
concatenated_df['agreement'] = concatenated_df['gpt_biome'] == concatenated_df['biome']

# Assuming concatenated_df is your DataFrame

# Prepare data for plotting - only selecting 'True' (correct) agreements
correct_data = concatenated_df[concatenated_df['agreement'] == True].groupby(['label']).size()

# Calculate total counts for each label to use for percentage calculation
total_counts = concatenated_df.groupby(['label']).size()

# Convert counts to percentages
correct_percentages = (correct_data / total_counts) * 100

# Plot
plt.figure(figsize=(10, 6))
ax = correct_percentages.plot(kind='bar', color='green', figsize=(10, 6))
plt.title('Percentage of correct gpt output')
plt.ylabel('agreement (%)')
plt.xlabel('distinguishing feature(s)')
plt.xticks(rotation=45)
plt.tight_layout()

# Annotate bars with percentages and counts
for idx, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    label = correct_percentages.index[idx]  # Use the index label directly
    count = correct_data[label]  # Get the count for the "correct" bar using the label
    
    if height > 0:  # Avoid annotating zero-height bars
        ax.text(x + width/2, y + height + 1, f'{height:.1f}%\n(n={count})', ha='center', va='center')

plt.show()











# Calculate agreement for each biome within each label
agreement_by_biome = concatenated_df.groupby(['label', 'biome', 'agreement']).size().unstack(fill_value=0).reset_index()

# If necessary, you can rename the columns for clarity
agreement_by_biome.columns = ['Label', 'Biome', 'Disagreement', 'Agreement']

##########


##########

# big plot - separated by biome 

# Unique labels and biomes for plotting
unique_labels = agreement_by_biome['Label'].unique()

# Convert the 'Biome' column to a list of unique biomes if it's not already
unique_biomes = agreement_by_biome['Biome'].unique().tolist()

# Create a dictionary to map each label to a numerical index
label_positions = {label: idx for idx, label in enumerate(agreement_by_biome['Label'].unique())}

# Set up the figure
plt.figure(figsize=(12, 8))

# Define a small font size for the annotations
annotation_font_size = 6

# Plot each bar with annotations
for idx, row in agreement_by_biome.iterrows():
    label_pos = label_positions[row['Label']]
    biome_offset = unique_biomes.index(row['Biome'])

    # Compute the base position for the current bar
    base_position = label_pos * (len(unique_biomes) + 1) + biome_offset

    # Plot the 'Agreement' part of the bar
    agreement_bar = plt.bar(base_position, row['Agreement'], color='green', edgecolor='white', width=0.8, label='Agreement' if idx == 0 else "")
    
    # Plot the 'Disagreement' part of the bar, stacked on the 'Agreement' part
    disagreement_bar = plt.bar(base_position, row['Disagreement'], bottom=row['Agreement'], color='red', edgecolor='white', width=0.8, label='Disagreement' if idx == 0 else "")

    # Annotate the bar with the biome name
    plt.text(base_position, 0, row['Biome'], rotation=90, va='bottom', ha='center', fontsize=annotation_font_size)

# Finalize the plot
plt.xlabel('')
plt.ylabel('# samples')
plt.title('agreement GPT output with ground truth')
plt.xticks(ticks=np.arange(len(label_positions)) * (len(unique_biomes) + 1) + len(unique_biomes)/2, labels=list(label_positions.keys()), rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

plt.tight_layout()
plt.show()

##########






































# older code: 

def calculate_agreement(merged_df):
    total_samples = len(merged_df)
    agree_samples = len(merged_df[merged_df['gpt_generated_output_clean'] == merged_df['biome']])
    agreement = (agree_samples / total_samples) * 100
    return agreement, total_samples




# -----------------------------
# 6. Dataframe Processing & Agreement Calculation for Overall Plot (All Samples)
# -----------------------------
# Creating a dataframe for the overall agreement and sample size per label (all samples)
overall_agreement_df_all = pd.DataFrame(columns=['label', 'agreement_percentage', 'total_samples'])

# Create a list of tuples mapping each file to its label
file_label_mapping = [(file, extract_labels_from_filename(file, distinguishing_features)) for file in selected_files]

# Sort the mapping based on labels
file_label_mapping.sort(key=lambda x: custom_sort(x[1]))

# Use the mapping for data processing
for file, label in file_label_mapping:
    processed_df = load_and_process_file(file, gold_dict_df)  # No common samples filter applied
    agreement, samples = calculate_agreement(processed_df)
    new_row = pd.DataFrame({
        'label': [label],
        'agreement_percentage': [agreement],
        'total_samples': [samples]
    })
    overall_agreement_df_all = pd.concat([overall_agreement_df_all, new_row], ignore_index=True)



# -----------------------------
# 6. Dataframe Processing & Agreement Calculation for Overall Plot (common samples)
# -----------------------------



all_samples = []
for file in selected_files:
    # Load the CSV file without headers
    temp_df = pd.read_csv(file, header=None)

    # Use .iloc to select the first column (assuming 'sample' is always the first column)
    sample_column = temp_df.iloc[:, 0]

    # Append the set of samples to the list
    all_samples.append(set(sample_column))


common_samples = set.intersection(*all_samples)



# Initialize the dataframe for overall agreement (common samples only)
overall_agreement_df_common = pd.DataFrame(columns=['label', 'agreement_percentage', 'total_samples'])


def filter_common_samples(df, common_samples):
    return df[df['sample'].isin(common_samples)]


# Use the mapping for data processing with common samples filter
for file, label in file_label_mapping:
    processed_df = load_and_process_file(file, gold_dict_df)
    processed_df_common = filter_common_samples(processed_df, common_samples)  # Apply common samples filter
    agreement, samples = calculate_agreement(processed_df_common)
    new_row = pd.DataFrame({
        'label': [label],
        'agreement_percentage': [agreement],
        'total_samples': [samples]
    })
    overall_agreement_df_common = pd.concat([overall_agreement_df_common, new_row], ignore_index=True)

# Sort the dataframe by label
overall_agreement_df_common['label'] = sorted(overall_agreement_df_common['label'], key=custom_sort)



# -----------------------------
# 7. Plotting & Visualization for Overall Plot (all + common samples)
# -----------------------------

# Add a new column to each DataFrame to distinguish between 'All Samples' and 'Common Samples'
overall_agreement_df_all['sample_type'] = 'all_samples'
overall_agreement_df_common['sample_type'] = 'common_samples'

# Merge the two DataFrames
combined_agreement_df = pd.concat([overall_agreement_df_all, overall_agreement_df_common])

# Sort the combined dataframe by label
combined_agreement_df.sort_values(by='label', inplace=True)

combined_agreement_df['sub_label'] = combined_agreement_df.apply(lambda x: f"{x['agreement_percentage']:.2f}% \n(n={x['total_samples']})", axis=1)


# -----------------------------
# 8. Combined Plotting & Visualization
# -----------------------------

# Function to extract numeric value from the label: useful for sorting
def extract_number(label):
    number = re.findall(r'\d+', label)
    return int(number[0]) if number else 0

# Add a numeric column for sorting
combined_agreement_df['numeric_label'] = combined_agreement_df['label'].apply(extract_number)

# Sort the DataFrame based on the numeric label
combined_agreement_df.sort_values('numeric_label', inplace=True)



plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

# Create a grouped bar plot
bar_plot_combined = sns.barplot(data=combined_agreement_df, x='label', y='agreement_percentage', 
                                hue='sample_type', palette="viridis")

# Adjust legend, title, and axis labels
bar_plot_combined.set_title('Agreement Percentage per Label for All vs Common Samples')
bar_plot_combined.set_ylabel('Agreement Percentage')

# Annotate bars with combined agreement percentage and total samples
num_bars = len(combined_agreement_df['label'].unique())
num_types = len(combined_agreement_df['sample_type'].unique())

for index, p in enumerate(bar_plot_combined.patches):
    height = p.get_height()
    # Calculate the index of the data row that corresponds to this patch
    data_row_index = (index % num_bars) * num_types + (index // num_bars)
    label_row = combined_agreement_df.iloc[data_row_index]
    annotation = label_row['sub_label']

    bar_plot_combined.annotate(annotation, (p.get_x() + p.get_width() / 2, height), 
                               ha='center', va='bottom', fontsize=10)

# Move legend outside of the plot
plt.legend(title='Sample Type', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.show()


































# -----------------------------
# 8. Plot confusion matrix files vs gold_dict 
# -----------------------------

def create_confusion_matrix(file, df_predicted_biomes):
    df = load_and_process_file(file, df_predicted_biomes)
    labels = ['animal', 'water', 'plant', 'soil', 'unknown']

    # Generate confusion matrix
    cm = confusion_matrix(df['biome'], df['gpt_generated_output_clean'], labels=labels)

    # Calculate counts for each category
    actual_counts = df['biome'].value_counts().reindex(labels, fill_value=0)
    predicted_counts = df['gpt_generated_output_clean'].value_counts().reindex(labels, fill_value=0)

    return cm, actual_counts, predicted_counts

def plot_multiple_confusion_matrices(confusion_matrices):
    # Determine the layout based on the number of matrices
    num_matrices = len(confusion_matrices)
    if num_matrices <= 2:
        nrows, ncols = 1, num_matrices
        title_font_size = 12  # Larger font for fewer matrices
    else:
        nrows = 2
        ncols = (num_matrices + 1) // 2  # Round up if odd number of matrices
        title_font_size = 8  # Smaller font for more matrices

    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
    axes = axes.flatten()  # Flatten in case of single row

    # Plot each confusion matrix
    for ax, (cm, actual_counts, predicted_counts, title) in zip(axes, confusion_matrices):
        xticklabels = [f'{label} ({predicted_counts[label]})' for label in actual_counts.index]
        yticklabels = [f'{label} ({actual_counts[label]})' for label in predicted_counts.index]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
        wrapped_title = "\n".join(textwrap.wrap(title, 60))
        ax.set_title(wrapped_title, fontsize=title_font_size)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    # Adjust layout
    plt.tight_layout()
    plt.show()


# GPT vs gold_dict
confusion_matrices_gold_dict = []
for file in selected_files:
    cm, actual_counts, predicted_counts = create_confusion_matrix(file, gold_dict_df)
    title = os.path.basename(file)
    confusion_matrices_gold_dict.append((cm, actual_counts, predicted_counts, title))
plot_multiple_confusion_matrices(confusion_matrices_gold_dict)

# GPT vs Joao
joao_biomes_path = os.path.join(work_dir, "joao_biomes_parsed.csv")

joao_biomes_df = pd.read_csv(joao_biomes_path)
joao_biomes_df['biome'] = joao_biomes_df['biome'].fillna('unknown').replace('aquatic', 'water').astype(str)
joao_biomes_all = joao_biomes_df.copy()
joao_biomes_high_confidence = joao_biomes_df[joao_biomes_df['confidence'] != 'low']

confusion_matrices_joao = []
for file in selected_files:
    cm, actual_counts, predicted_counts = create_confusion_matrix(file, joao_biomes_all)
    title = os.path.basename(file)
    confusion_matrices_joao.append((cm, actual_counts, predicted_counts, title))
plot_multiple_confusion_matrices(confusion_matrices_joao)







# -----------------------------
# Step 1: Extract Samples from the Chosen GPT File
# -----------------------------

chosen_file = selected_files[3]  # You can modify this to select a specific file

df_chosen = pd.read_csv(chosen_file)
chosen_samples = set(df_chosen['sample'])


# -----------------------------
# Step 2: Load Joao's Biomes Data
# -----------------------------
joao_biomes_path = os.path.join(work_dir, "joao_biomes_parsed.csv")
joao_biomes_df = pd.read_csv(joao_biomes_path)
joao_biomes_df['biome'] = joao_biomes_df['biome'].fillna('unknown').replace('aquatic', 'water').astype(str)


# -----------------------------
# Step 3: Filter Joao's and Gold Dict Biomes Dataframes
# -----------------------------
filtered_joao_df = joao_biomes_df[joao_biomes_df['sample'].isin(chosen_samples)]
filtered_gold_dict_df = gold_dict_df[gold_dict_df['sample'].isin(chosen_samples)]


# -----------------------------
# Step 4: Function for Comparing Two Biome Datasets with Sample Sums
# -----------------------------
def create_and_plot_confusion_matrix(df_joao, df_gold_dict):
    # Merge the two dataframes on the common 'sample' column
    merged_df = df_joao.merge(df_gold_dict, on='sample', suffixes=('_joao', '_gold'))
    
    # Define the labels for the confusion matrix
    labels = ['animal', 'water', 'plant', 'soil', 'unknown']

    # Count the number of samples for each biome in both datasets
    joao_counts = merged_df['biome_joao'].value_counts().reindex(labels, fill_value=0)
    gold_dict_counts = merged_df['biome_gold'].value_counts().reindex(labels, fill_value=0)

    # Generate the confusion matrix
    cm = confusion_matrix(merged_df['biome_gold'], merged_df['biome_joao'], labels=labels)

    # Create labels with sample counts
    xticklabels = [f'{label} ({joao_counts[label]})' for label in labels]
    yticklabels = [f'{label} ({gold_dict_counts[label]})' for label in labels]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title('Confusion Matrix: Gold Dict Biomes vs Joao\'s Biomes')
    plt.ylabel('Gold Dict Biomes')
    plt.xlabel('Joao\'s Biomes')
    plt.show()


# -----------------------------
# Step 5: Create and Plot the Confusion Matrix
# -----------------------------
create_and_plot_confusion_matrix(filtered_joao_df, filtered_gold_dict_df)