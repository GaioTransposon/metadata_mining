#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:05:41 2023

@author: dgaio
"""

# run as: 
# python ~/github/metadata_mining/scripts/compare_gpt_outputs.py --work_dir MicrobeAtlasProject



import os
import re
import pandas as pd
import pickle

def load_gold_standard(filepath):
    """ Load the gold standard dictionary from a pickle file. """
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def process_gpt_file(file_path, gold_dict_df):
    """ Process each GPT output file to calculate agreement with gold standard. """
    try:
        gpt_data = pd.read_csv(file_path, usecols=[0, 1], header=None, names=['sample', 'biome'])
        merged_df = gpt_data.merge(gold_dict_df, on='sample', suffixes=('_gpt', '_gold'))
        merged_df['agreement'] = merged_df['biome_gpt'] == merged_df['biome_gold']
        agreement_ratio = merged_df['agreement'].mean()
        valid_samples = merged_df['agreement'].count()  # Count only the matched samples
        return [agreement_ratio, valid_samples]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [None, 0]

def parse_filename(filename):
    """ Extract details from the filename using regex. """
    pattern = (
        r"gpt_clean_output_"
        r"(?P<nspb>nspb\d+)_"
        r"chunking(?P<chunking>yes|no)_"
        r"chunksize(?P<chunksize>\d+)_"
        r"model(?P<model>[\w.-]+)_"
        r"temp(?P<temp>\d\.\d)_"
        r"maxtokens(?P<maxtokens>\d+)_"
        r"topp(?P<topp>\d\.\d+)_"
        r"freqp(?P<freqp>\d\.\d+)_"
        r"presp(?P<presp>\d\.\d+)_"
        r"rs(?P<rs>\d+)_"
        r"(?P<miscellaneous>.*?)_"
        r"dt(?P<dt>\d{12})"
        r"(?:\.(txt|csv))"
    )
    match = re.match(pattern, filename)
    return match.groupdict() if match else None

def main():
    work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
    home_dir = "/Users/dgaio"
    gold_dict_path = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
    gold_dict = load_gold_standard(gold_dict_path)
    gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
    gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
    gold_dict_df.drop(columns='tuple_data', inplace=True)

    results_dict = {}
    gpt_files_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject"
    for filename in os.listdir(gpt_files_dir):
        if filename.startswith("gpt_clean_output_nspb"):
            file_path = os.path.join(gpt_files_dir, filename)
            results_dict[filename] = process_gpt_file(file_path, gold_dict_df)

    parsed_filenames = []
    for filename, results in results_dict.items():
        file_details = parse_filename(filename)
        if file_details:
            file_details.update({"filename": filename, "agreement_ratio": results[0], "sample_size": results[1]})
            parsed_filenames.append(file_details)

    results_df = pd.DataFrame(parsed_filenames)
    print(results_df.head())

    output_path = os.path.join(work_dir, 'gpt_outputs_biome_agreement.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
















# -----------------------------
# 0. Imports
# -----------------------------
import glob
import pandas as pd
import os
import pickle
import re
import matplotlib.pyplot as plt

# -----------------------------
# 1. Paths
# -----------------------------

home_dir = os.getenv('HOME')
work_dir = "MicrobeAtlasProject"

# -----------------------------
# 2. Function Definitions
# -----------------------------

def interactive_file_selection(initial_pattern, work_dir):
    """
    Allow user to interactively select files based on their filename patterns or indices.
    """
    
    if not os.path.isabs(work_dir):
        work_dir = os.path.abspath(work_dir)
        
    if not os.path.exists(work_dir):
        print(f"The specified working directory does not exist: {work_dir}")
        return []
    
    full_pattern = os.path.join(work_dir, initial_pattern)
    
        
    print("Current Working Directory:", work_dir)  
    print("Full Pattern:", full_pattern)  

    current_files = glob.glob(full_pattern)
    
    
    selected_files = []  
    print("Files found:", current_files)
    
    
    while True:
        print("\nCurrent matching files:")
        for idx, file in enumerate(current_files, start=1):
            print(f"{idx}. {os.path.basename(file)}")

        action = input("\nEnter a keyword to refine further (must start with string), 'done' to finish, 'all' to select all, or space-separated indices (e.g., '2 3') to select specific files: ").strip().lower()
        
        if action == 'done':
            return selected_files
        elif action == 'all':
            return current_files
        else:
            indices = action.split()
            if all(idx.isdigit() for idx in indices):  
                indices = [int(idx) - 1 for idx in indices]  # Convert to 0-based index
                selected_files = [current_files[idx] for idx in indices if 0 <= idx < len(current_files)]  # check if index is in range
                return selected_files
            else:
                current_files = [f for f in current_files if action in f]
                
                if not current_files:
                    print("No files match your refined criteria. Resetting to initial files.")
                    current_files = glob.glob(initial_pattern)



def find_distinguishing_features(files):
    """
    Determine the distinguishing features between filenames.
    Collect all tokens and identify those that are unique to some but not all filenames.
    """
    all_tokens = []
    file_tokens = []

    for file in files:
        tokens = os.path.basename(file).split('_')[:-2]  # to exclude date and time
        file_tokens.append(set(tokens))
        all_tokens.extend(tokens)

    token_count = {}
    for token in set(all_tokens):
        token_count[token] = sum(1 for tokens in file_tokens if token in tokens)

    # find tokens that are unique to some files but not common to all
    num_files = len(files)
    distinguishing_tokens = {token for token, count in token_count.items() if count != num_files}

    return distinguishing_tokens



def extract_labels_from_filename(filename, distinguishing_tokens):
    """
    Extract distinguishing labels from the filename.
    """
    # extract content between "gpt" and "_dt"
    pattern = re.compile(r'gpt(.*?)_dt')
    matches = pattern.search(filename)

    if matches:
        content = matches.group(1)

        labels = content.split('_')
        labels = [label for label in labels if label in distinguishing_tokens]
        
        return ", ".join(labels)

    return "Unknown"



def load_and_process_file(file_name, gold_standard_df, label):
    dfr = pd.read_csv(file_name, header=None) # without headers, as the number of columns can vary

    dfr = dfr.iloc[:, [0, 1]]
    dfr.columns = ['sample', 'gpt_biome']
    dfr['label'] = label

    merged_df = pd.merge(dfr, gold_standard_df, on='sample', how='inner')

    return merged_df


def edit_features(file_label_map):
    """
    Allow the user to edit each label extracted from filenames, maintaining the dictionary structure.
    """
    print("Current labels for each file:")
    for idx, (file, label) in enumerate(file_label_map.items(), start=1):
        print(f"{idx}. {os.path.basename(file)} - {label}")

    if input("Do you want to edit any labels? (y/n): ").strip().lower() == 'y':
        for file in list(file_label_map.keys()):  
            current_label = file_label_map[file]
            new_label = input(f"Change the label for '{os.path.basename(file)}' from '{current_label}' to (press enter to keep the same): ")
            if new_label:
                file_label_map[file] = new_label

    return file_label_map


def match_files_to_features(files, labels):
    file_feature_map = {}
    for file in files:
        file_base_name = os.path.basename(file)
        # Check if any label fully matches sections of the filename
        for label in labels:
            label_parts = label.split(", ")
            if all(part.strip() in file_base_name for part in label_parts):
                file_feature_map[file] = label
                break
    return file_feature_map


# -----------------------------
# 3. Data Processing & Loading
# -----------------------------    
input_gold_dict = os.path.join(home_dir, "github/metadata_mining/source_data/gold_dict.pkl")
with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)


# -----------------------------
# 4. Interactive File Selection & feature extraction
# -----------------------------
    
initial_pattern = "gpt_clean*"
selected_files = interactive_file_selection(initial_pattern, work_dir)
print("\nSelected files for analysis:")
for f in selected_files:
    print(os.path.basename(f))
    
distinguishing_features = find_distinguishing_features(selected_files)
#352 367

file_label_map = {file: extract_labels_from_filename(file, distinguishing_features) for file in selected_files}

file_label_map = edit_features(file_label_map)


all_dfs = []
for file, label in file_label_map.items():
    print('\n')
    print(file)
    print(label, '\n')
    processed_df = load_and_process_file(file, gold_dict_df, label)
    all_dfs.append(processed_df)

concatenated_df = pd.concat(all_dfs, ignore_index=True)
concatenated_df['agreement'] = concatenated_df['gpt_biome'] == concatenated_df['biome']



# -----------------------------
# 5. Calculating agreement and plotting
# -----------------------------

############

# check if count of True per label matches the plot:
df = concatenated_df
df['agreement'] = df['agreement'].astype(int)  
true_counts = df.groupby('label')['agreement'].sum()
total_counts = df.groupby('label').size()
true_percentage = (true_counts / total_counts * 100).round(2)  
result = pd.DataFrame({
    'True Counts': true_counts,
    'Total Counts': total_counts,
    'Percentage True': true_percentage
})

############

# only for plot name: 
unique_parts = set()
for label in file_label_map.values():
    parts = label.split(', ')  # split each label into parts
    filtered_parts = [part for part in parts if 'API' not in part]  # filter out parts containing 'API'
    unique_parts.update(filtered_parts)

# Join the unique parts sorted alphabetically and separated by '_'
feature_description = "_".join(sorted(unique_parts))

# Plot
plt.figure(figsize=(10, 6))
ax = result['Percentage True'].plot(kind='bar', color='green')
plt.title('Percentage of correct GPT output')
plt.ylabel('agreement (%)')
plt.xlabel('distinguishing feature(s)')
plt.xticks(rotation=45)  

for idx, p in enumerate(ax.patches):
    height = p.get_height()
    x, y = p.get_xy()
    label = result.index[idx]
    total_count = result.at[label, 'Total Counts']
    percentage = result.at[label, 'Percentage True']
    
    if height > 0:  # avoid annotating zero-height bars
        ax.text(x + p.get_width() / 2, y + height + 1, f'{percentage}%\n(n={total_count})', ha='center', va='center')

plt.tight_layout()
plot_filename = os.path.join(work_dir, f'agreement_{feature_description}.png')
#plt.savefig(plot_filename)
plt.show()
#plt.close()
print(f"Plot saved as: {plot_filename}")


############


filenames = [os.path.basename(file) for file in selected_files]
for i in filenames: 
    print(i)

print(result)

# =============================================================================
# 
# # Generate filename based on distinguishing features
# feature_description = "_".join(sorted(distinguishing_features)).replace(" ", "_")
# plot_filename = os.path.join(work_dir, f'agreement_by_biome_{feature_description}.png')
# plt.savefig(plot_filename)
# # plt.show()
# plt.close()
# 
# print(f"Plot saved as: {plot_filename}")
# =============================================================================

############



import scipy.stats as stats

# Assuming 'concatenated_df' has a column 'label' for group labels and 'agreement' as binary or continuous data
grouped_data = {label: df['agreement'].tolist() for label, df in concatenated_df.groupby('label')}

# Perform ANOVA
f_value, p_value = stats.f_oneway(*grouped_data.values())
print("ANOVA test results: F-value = {:.2f}, p-value = {:.5f}".format(f_value, p_value))

# Adding this to decide on reporting the result
if p_value < 0.05:
    print("Statistically significant differences exist between the groups.")
else:
    print("No statistically significant differences were found between the groups.")

















import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'work_dir' is already defined and 'selected_files' are available
joao_biomes_path = os.path.join(work_dir, "joao_biomes_parsed.csv")
joao_biomes_df = pd.read_csv(joao_biomes_path)
joao_biomes_df['biome'] = joao_biomes_df['biome'].fillna('unknown').replace('aquatic', 'water').astype(str)
joao_biomes_all = joao_biomes_df.copy()
#joao_biomes_high_confidence = joao_biomes_df[joao_biomes_df['confidence'] != 'low']



# Filter concatenated_df for rows where the label equals 'async'
filtered_concat_df = concatenated_df[concatenated_df['label'] == 'sync']


filtered_concat_df
gold_dict_df
joao_biomes_all



# Extract sample IDs from each DataFrame
gpt_samples = set(filtered_concat_df['sample'])
joao_samples = set(joao_biomes_all['sample'])
gold_samples = set(gold_dict_df['sample'])

# Find common sample IDs among all three DataFrames
common_samples = gpt_samples.intersection(joao_samples, gold_samples)
print(f"Number of common samples: {len(common_samples)}")


# Filter each DataFrame to include only common samples
filtered_concat_df = filtered_concat_df[filtered_concat_df['sample'].isin(common_samples)]
len(filtered_concat_df)
filtered_joao_df = joao_biomes_all[joao_biomes_all['sample'].isin(common_samples)]
len(filtered_joao_df)
filtered_gold_df = gold_dict_df[gold_dict_df['sample'].isin(common_samples)]
len(filtered_gold_df)


# Sort DataFrames by 'sample' to align them
filtered_concat_df = filtered_concat_df.sort_values(by='sample').reset_index(drop=True)
filtered_joao_df = filtered_joao_df.sort_values(by='sample').reset_index(drop=True)
filtered_gold_df = filtered_gold_df.sort_values(by='sample').reset_index(drop=True)

# Combine all unique labels from GPT, João, and the Gold Standard
all_labels = np.union1d(filtered_concat_df['gpt_biome'].unique(), filtered_gold_df['biome'].unique())
all_labels = np.union1d(all_labels, filtered_joao_df['biome'].unique())


# Compute the initial confusion matrices with all labels included
cm_gpt_gold = confusion_matrix(filtered_gold_df['biome'], filtered_concat_df['gpt_biome'], labels=all_labels)
cm_joao_gold = confusion_matrix(filtered_gold_df['biome'], filtered_joao_df['biome'], labels=all_labels)

# =============================================================================
# def filter_zero_rows_columns(cm, labels):
#     # Determine which rows and columns sum to zero
#     row_sums = cm.sum(axis=1)
#     col_sums = cm.sum(axis=0)
# 
#     # Filter out rows and columns where the sum is zero
#     non_zero_rows = row_sums != 0
#     non_zero_cols = col_sums != 0
# 
#     # Filter the confusion matrix and labels
#     filtered_cm = cm[non_zero_rows][:, non_zero_cols]
#     filtered_labels = labels[non_zero_cols]  # Only need to filter columns for labels as they are the predictions
# 
#     return filtered_cm, filtered_labels
# =============================================================================

def filter_zero_rows_columns(cm, labels):
    # Determine which rows and columns sum to zero
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    # Filter out rows and columns where the sum is zero
    non_zero_rows = row_sums != 0
    non_zero_cols = col_sums != 0

    # Create a mask for rows and columns that are non-zero
    mask = non_zero_rows[:, np.newaxis] & non_zero_cols[np.newaxis, :]

    # Apply mask to filter the confusion matrix
    filtered_cm = cm[mask].reshape(sum(non_zero_rows), sum(non_zero_cols))

    # Adjust labels for rows and columns
    filtered_row_labels = labels[non_zero_rows]
    filtered_col_labels = labels[non_zero_cols]

    return filtered_cm, filtered_row_labels, filtered_col_labels




# =============================================================================
# # Filter GPT vs Gold Standard matrix
# filtered_cm_gpt_gold, filtered_labels_gpt_gold = filter_zero_rows_columns(cm_gpt_gold, all_labels)
# 
# # Filter João vs Gold Standard matrix
# filtered_cm_joao_gold, filtered_labels_joao_gold = filter_zero_rows_columns(cm_joao_gold, all_labels)
# 
# =============================================================================

# =============================================================================
# def plot_confusion_matrix(cm, labels, title):
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(title)
#     plt.show()
# =============================================================================

def plot_confusion_matrix(cm, row_labels, col_labels, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=col_labels, yticklabels=row_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()



# =============================================================================
# # Plotting the filtered confusion matrices
# plot_confusion_matrix(filtered_cm_gpt_gold, filtered_labels_gpt_gold, "Filtered Confusion Matrix: GPT vs Gold Standard")
# plot_confusion_matrix(filtered_cm_joao_gold, filtered_labels_joao_gold, "Filtered Confusion Matrix: João vs Gold Standard")
# 
# 
# =============================================================================

# Filter GPT vs Gold Standard matrix
filtered_cm_gpt_gold, row_labels_gpt_gold, col_labels_gpt_gold = filter_zero_rows_columns(cm_gpt_gold, all_labels)

# Filter João vs Gold Standard matrix
filtered_cm_joao_gold, row_labels_joao_gold, col_labels_joao_gold = filter_zero_rows_columns(cm_joao_gold, all_labels)

# Plotting the filtered confusion matrices
plot_confusion_matrix(filtered_cm_gpt_gold, row_labels_gpt_gold, col_labels_gpt_gold, "Filtered Confusion Matrix: GPT vs Gold Standard")
plot_confusion_matrix(filtered_cm_joao_gold, row_labels_joao_gold, col_labels_joao_gold, "Filtered Confusion Matrix: João vs Gold Standard")


# =============================================================================
# # older code: 
# 
# def calculate_agreement(merged_df):
#     total_samples = len(merged_df)
#     agree_samples = len(merged_df[merged_df['gpt_generated_output_clean'] == merged_df['biome']])
#     agreement = (agree_samples / total_samples) * 100
#     return agreement, total_samples
# 
# 
# 
# 
# # -----------------------------
# # 6. Dataframe Processing & Agreement Calculation for Overall Plot (All Samples)
# # -----------------------------
# # Creating a dataframe for the overall agreement and sample size per label (all samples)
# overall_agreement_df_all = pd.DataFrame(columns=['label', 'agreement_percentage', 'total_samples'])
# 
# # Create a list of tuples mapping each file to its label
# file_label_mapping = [(file, extract_labels_from_filename(file, distinguishing_features)) for file in selected_files]
# 
# # Sort the mapping based on labels
# file_label_mapping.sort(key=lambda x: custom_sort(x[1]))
# 
# # Use the mapping for data processing
# for file, label in file_label_mapping:
#     processed_df = load_and_process_file(file, gold_dict_df)  # No common samples filter applied
#     agreement, samples = calculate_agreement(processed_df)
#     new_row = pd.DataFrame({
#         'label': [label],
#         'agreement_percentage': [agreement],
#         'total_samples': [samples]
#     })
#     overall_agreement_df_all = pd.concat([overall_agreement_df_all, new_row], ignore_index=True)
# 
# 
# 
# # -----------------------------
# # 6. Dataframe Processing & Agreement Calculation for Overall Plot (common samples)
# # -----------------------------
# 
# 
# 
# all_samples = []
# for file in selected_files:
#     # Load the CSV file without headers
#     temp_df = pd.read_csv(file, header=None)
# 
#     # Use .iloc to select the first column (assuming 'sample' is always the first column)
#     sample_column = temp_df.iloc[:, 0]
# 
#     # Append the set of samples to the list
#     all_samples.append(set(sample_column))
# 
# 
# common_samples = set.intersection(*all_samples)
# 
# 
# 
# # Initialize the dataframe for overall agreement (common samples only)
# overall_agreement_df_common = pd.DataFrame(columns=['label', 'agreement_percentage', 'total_samples'])
# 
# 
# def filter_common_samples(df, common_samples):
#     return df[df['sample'].isin(common_samples)]
# 
# 
# # Use the mapping for data processing with common samples filter
# for file, label in file_label_mapping:
#     processed_df = load_and_process_file(file, gold_dict_df)
#     processed_df_common = filter_common_samples(processed_df, common_samples)  # Apply common samples filter
#     agreement, samples = calculate_agreement(processed_df_common)
#     new_row = pd.DataFrame({
#         'label': [label],
#         'agreement_percentage': [agreement],
#         'total_samples': [samples]
#     })
#     overall_agreement_df_common = pd.concat([overall_agreement_df_common, new_row], ignore_index=True)
# 
# # Sort the dataframe by label
# overall_agreement_df_common['label'] = sorted(overall_agreement_df_common['label'], key=custom_sort)
# 
# 
# 
# # -----------------------------
# # 7. Plotting & Visualization for Overall Plot (all + common samples)
# # -----------------------------
# 
# # Add a new column to each DataFrame to distinguish between 'All Samples' and 'Common Samples'
# overall_agreement_df_all['sample_type'] = 'all_samples'
# overall_agreement_df_common['sample_type'] = 'common_samples'
# 
# # Merge the two DataFrames
# combined_agreement_df = pd.concat([overall_agreement_df_all, overall_agreement_df_common])
# 
# # Sort the combined dataframe by label
# combined_agreement_df.sort_values(by='label', inplace=True)
# 
# combined_agreement_df['sub_label'] = combined_agreement_df.apply(lambda x: f"{x['agreement_percentage']:.2f}% \n(n={x['total_samples']})", axis=1)
# 
# 
# # -----------------------------
# # 8. Combined Plotting & Visualization
# # -----------------------------
# 
# # Function to extract numeric value from the label: useful for sorting
# def extract_number(label):
#     number = re.findall(r'\d+', label)
#     return int(number[0]) if number else 0
# 
# # Add a numeric column for sorting
# combined_agreement_df['numeric_label'] = combined_agreement_df['label'].apply(extract_number)
# 
# # Sort the DataFrame based on the numeric label
# combined_agreement_df.sort_values('numeric_label', inplace=True)
# 
# 
# 
# plt.figure(figsize=(15, 8))
# sns.set_style("whitegrid")
# 
# # Create a grouped bar plot
# bar_plot_combined = sns.barplot(data=combined_agreement_df, x='label', y='agreement_percentage', 
#                                 hue='sample_type', palette="viridis")
# 
# # Adjust legend, title, and axis labels
# bar_plot_combined.set_title('Agreement Percentage per Label for All vs Common Samples')
# bar_plot_combined.set_ylabel('Agreement Percentage')
# 
# # Annotate bars with combined agreement percentage and total samples
# num_bars = len(combined_agreement_df['label'].unique())
# num_types = len(combined_agreement_df['sample_type'].unique())
# 
# for index, p in enumerate(bar_plot_combined.patches):
#     height = p.get_height()
#     # Calculate the index of the data row that corresponds to this patch
#     data_row_index = (index % num_bars) * num_types + (index // num_bars)
#     label_row = combined_agreement_df.iloc[data_row_index]
#     annotation = label_row['sub_label']
# 
#     bar_plot_combined.annotate(annotation, (p.get_x() + p.get_width() / 2, height), 
#                                ha='center', va='bottom', fontsize=10)
# 
# # Move legend outside of the plot
# plt.legend(title='Sample Type', bbox_to_anchor=(1.01, 1), loc='upper left')
# 
# plt.tight_layout()
# plt.show()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # -----------------------------
# # 8. Plot confusion matrix files vs gold_dict 
# # -----------------------------
# 
# def create_confusion_matrix(file, df_predicted_biomes):
#     df = load_and_process_file(file, df_predicted_biomes)
#     labels = ['animal', 'water', 'plant', 'soil', 'unknown']
# 
#     # Generate confusion matrix
#     cm = confusion_matrix(df['biome'], df['gpt_generated_output_clean'], labels=labels)
# 
#     # Calculate counts for each category
#     actual_counts = df['biome'].value_counts().reindex(labels, fill_value=0)
#     predicted_counts = df['gpt_generated_output_clean'].value_counts().reindex(labels, fill_value=0)
# 
#     return cm, actual_counts, predicted_counts
# 
# def plot_multiple_confusion_matrices(confusion_matrices):
#     # Determine the layout based on the number of matrices
#     num_matrices = len(confusion_matrices)
#     if num_matrices <= 2:
#         nrows, ncols = 1, num_matrices
#         title_font_size = 12  # Larger font for fewer matrices
#     else:
#         nrows = 2
#         ncols = (num_matrices + 1) // 2  # Round up if odd number of matrices
#         title_font_size = 8  # Smaller font for more matrices
# 
#     fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
#     axes = axes.flatten()  # Flatten in case of single row
# 
#     # Plot each confusion matrix
#     for ax, (cm, actual_counts, predicted_counts, title) in zip(axes, confusion_matrices):
#         xticklabels = [f'{label} ({predicted_counts[label]})' for label in actual_counts.index]
#         yticklabels = [f'{label} ({actual_counts[label]})' for label in predicted_counts.index]
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
#         wrapped_title = "\n".join(textwrap.wrap(title, 60))
#         ax.set_title(wrapped_title, fontsize=title_font_size)
#         ax.set_ylabel('Actual')
#         ax.set_xlabel('Predicted')
# 
#     # Adjust layout
#     plt.tight_layout()
#     plt.show()
# 
# 
# # GPT vs gold_dict
# confusion_matrices_gold_dict = []
# for file in selected_files:
#     cm, actual_counts, predicted_counts = create_confusion_matrix(file, gold_dict_df)
#     title = os.path.basename(file)
#     confusion_matrices_gold_dict.append((cm, actual_counts, predicted_counts, title))
# plot_multiple_confusion_matrices(confusion_matrices_gold_dict)
# 
# # GPT vs Joao
# joao_biomes_path = os.path.join(work_dir, "joao_biomes_parsed.csv")
# 
# joao_biomes_df = pd.read_csv(joao_biomes_path)
# joao_biomes_df['biome'] = joao_biomes_df['biome'].fillna('unknown').replace('aquatic', 'water').astype(str)
# joao_biomes_all = joao_biomes_df.copy()
# joao_biomes_high_confidence = joao_biomes_df[joao_biomes_df['confidence'] != 'low']
# 
# confusion_matrices_joao = []
# for file in selected_files:
#     cm, actual_counts, predicted_counts = create_confusion_matrix(file, joao_biomes_all)
#     title = os.path.basename(file)
#     confusion_matrices_joao.append((cm, actual_counts, predicted_counts, title))
# plot_multiple_confusion_matrices(confusion_matrices_joao)
# 
# 
# 
# 
# 
# 
# 
# # -----------------------------
# # Step 1: Extract Samples from the Chosen GPT File
# # -----------------------------
# 
# chosen_file = selected_files[3]  # You can modify this to select a specific file
# 
# df_chosen = pd.read_csv(chosen_file)
# chosen_samples = set(df_chosen['sample'])
# 
# 
# # -----------------------------
# # Step 2: Load Joao's Biomes Data
# # -----------------------------
# joao_biomes_path = os.path.join(work_dir, "joao_biomes_parsed.csv")
# joao_biomes_df = pd.read_csv(joao_biomes_path)
# joao_biomes_df['biome'] = joao_biomes_df['biome'].fillna('unknown').replace('aquatic', 'water').astype(str)
# 
# 
# # -----------------------------
# # Step 3: Filter Joao's and Gold Dict Biomes Dataframes
# # -----------------------------
# filtered_joao_df = joao_biomes_df[joao_biomes_df['sample'].isin(chosen_samples)]
# filtered_gold_dict_df = gold_dict_df[gold_dict_df['sample'].isin(chosen_samples)]
# 
# 
# # -----------------------------
# # Step 4: Function for Comparing Two Biome Datasets with Sample Sums
# # -----------------------------
# def create_and_plot_confusion_matrix(df_joao, df_gold_dict):
#     # Merge the two dataframes on the common 'sample' column
#     merged_df = df_joao.merge(df_gold_dict, on='sample', suffixes=('_joao', '_gold'))
#     
#     # Define the labels for the confusion matrix
#     labels = ['animal', 'water', 'plant', 'soil', 'unknown']
# 
#     # Count the number of samples for each biome in both datasets
#     joao_counts = merged_df['biome_joao'].value_counts().reindex(labels, fill_value=0)
#     gold_dict_counts = merged_df['biome_gold'].value_counts().reindex(labels, fill_value=0)
# 
#     # Generate the confusion matrix
#     cm = confusion_matrix(merged_df['biome_gold'], merged_df['biome_joao'], labels=labels)
# 
#     # Create labels with sample counts
#     xticklabels = [f'{label} ({joao_counts[label]})' for label in labels]
#     yticklabels = [f'{label} ({gold_dict_counts[label]})' for label in labels]
# 
#     # Plot the confusion matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels)
#     plt.title('Confusion Matrix: Gold Dict Biomes vs Joao\'s Biomes')
#     plt.ylabel('Gold Dict Biomes')
#     plt.xlabel('Joao\'s Biomes')
#     plt.show()
# 
# 
# # -----------------------------
# # Step 5: Create and Plot the Confusion Matrix
# # -----------------------------
# create_and_plot_confusion_matrix(filtered_joao_df, filtered_gold_dict_df)
# =============================================================================


