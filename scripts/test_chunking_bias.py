#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:53:10 2024

@author: dgaio
"""


import pandas as pd
import os
import pickle

def process_files(metadata_file_path, biome_info_file_path):
    def parse_metadata(file_path):
        batch_number = 0
        data = []
        position_within_batch = 0

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if '-----' in line and position_within_batch > 0:
                    batch_number += 1
                    position_within_batch = 0
                elif 'sample_ID=' in line and '>' in line:
                    sample_id = line.split('>')[1].strip()
                    position_within_batch += 1
                    data.append((sample_id, f'batch {batch_number + 1}', position_within_batch))

        return pd.DataFrame(data, columns=['Sample ID', 'Batch Number', 'Position Within Batch'])

    def merge_with_biome_info(df_filtered, file_path_biome_info):
        df_biome_info = pd.read_csv(file_path_biome_info)
        df_merged = pd.merge(df_filtered, df_biome_info[['col_0', 'col_1']], left_on='Sample ID', right_on='col_0', how='left')
        df_merged.rename(columns={'col_1': 'Biome'}, inplace=True)
        df_merged.drop(columns=['col_0'], inplace=True)
        return df_merged

    # Parse the metadata file
    df_metadata = parse_metadata(metadata_file_path)

    # Filter for batches with 5 or more samples
    batch_counts = df_metadata['Batch Number'].value_counts()
    large_batches = batch_counts[batch_counts >= 5].index.tolist()
    df_filtered = df_metadata[df_metadata['Batch Number'].isin(large_batches)]

    # Merge with biome information and return the result
    df_final = merge_with_biome_info(df_filtered, biome_info_file_path)
    return df_final

# List of metadata and GPT output file pairs
file_pairs = [
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131529.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1536.txt"),
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131617.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1624.txt"),
    ("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/metadata_chunks_202403131733.txt", "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240313_1740.txt")
]

# Process each pair and concatenate the results
dfs = []
for metadata_file, biome_info_file in file_pairs:
    dfs.append(process_files(metadata_file, biome_info_file))


# Concatenate all resulting DataFrames from the processed file pairs
final_df = pd.concat(dfs, ignore_index=True)

# Display the final, comprehensive DataFrame
print(final_df)






# Load the gold dictionary
work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
input_gold_dict = os.path.join(work_dir, "gold_dict.pkl")

with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)

gold_dict = gold_dict[0]
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['sample', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['gold_biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])  # Use 'gold_biome' to differentiate
gold_dict_df.drop(columns='tuple_data', inplace=True)
# Optionally filter out unknown biomes
# gold_dict_df = gold_dict_df[gold_dict_df['gold_biome'] != 'unknown']

# Merge the final_df with gold_dict_df based on sample IDs
final_df_with_gold = pd.merge(final_df, gold_dict_df[['sample', 'gold_biome']], left_on='Sample ID', right_on='sample', how='left')
final_df_with_gold.drop(columns='sample', inplace=True)  # Drop the redundant 'sample' column after merge

# Display the final DataFrame with additional biome information from the gold dictionary
print(final_df_with_gold)



# Step 1: Determine Correctness
final_df_with_gold['Correct Guess'] = final_df_with_gold['Biome'] == final_df_with_gold['gold_biome']

# =============================================================================
# # Step 2: Categorize Sample Positions
# def categorize_position(row, n=1):  # You can adjust 'n' based on your batch sizes
#     if row['Position Within Batch'] <= n:
#         return 'beginning'
#     elif row['Position Within Batch'] > row['Batch Size'] - n:
#         return 'end'
#     else:
#         return 'middle'
# =============================================================================

def categorize_position(row):
    total = row['Batch Size']
    position = row['Position Within Batch']

    # Calculate the size of 'beginning' and 'end' segments
    # Ensuring that the 'middle' segment is as large as possible for odd-sized batches
    beginning_size = total // 3 if total % 3 == 0 else total // 3 + 1
    end_start = total - beginning_size + 1

    # Assign categories based on calculated sizes
    if position <= beginning_size:
        return 'beginning'
    elif position >= end_start:
        return 'end'
    else:
        return 'middle'




# Calculate batch sizes and add them to the DataFrame
batch_sizes = final_df_with_gold['Batch Number'].value_counts().rename('Batch Size')
final_df_with_gold = final_df_with_gold.merge(batch_sizes, left_on='Batch Number', right_index=True)

# Apply the categorization function
final_df_with_gold['Position Category'] = final_df_with_gold.apply(categorize_position, axis=1)





# Group the DataFrame by 'Position Category' and determine the minimum group size
min_group_size = final_df_with_gold.groupby('Position Category').size().min()

# Function to sample rows from each group
def sample_equal_number(group, seed):
    return group.sample(n=min_group_size, random_state=seed)  # Change the seed here for different random sampling

# Sample an equal number of rows from each 'Position Category' using a different seed
seed = 20  # Change this seed value each time you want to regenerate the plot with a different random sample
final_df_with_gold = final_df_with_gold.groupby('Position Category').apply(lambda x: sample_equal_number(x, seed)).reset_index(drop=True)



# Display the sampled DataFrame
print(final_df_with_gold)









# is this below right? 



import matplotlib.pyplot as plt

# Ensure the data is sorted by 'Position Category'
final_df_with_gold['Position Category'] = pd.Categorical(final_df_with_gold['Position Category'],
                                                         categories=['beginning', 'middle', 'end'],
                                                         ordered=True)

# Count the number of correct and incorrect guesses for each position category
position_correctness_counts = final_df_with_gold.groupby(['Position Category', 'Correct Guess']).size().unstack(fill_value=0)

# Print the sum of correct and incorrect guesses for verification
print("Counts of correct and incorrect guesses for each position category:")
print(position_correctness_counts)





# Calculate the total for each position category to use for percentage calculation
totals = position_correctness_counts.sum(axis=1)

# Calculate the percentage of correct and incorrect guesses within each category
position_correctness_percentages = position_correctness_counts.div(totals, axis=0) * 100

# Print the percentages for verification
print("\nPercentages of correct and incorrect guesses for each position category:")
print(position_correctness_percentages)





# labels not showing right 
# remove "other"? 

import matplotlib.pyplot as plt

# Count the number of correct and incorrect guesses for each position category
position_correctness_counts = final_df_with_gold.groupby(['Position Category', 'Correct Guess']).size().unstack(fill_value=0)

# Calculate the total for each position category to use for percentage calculation
totals = position_correctness_counts.sum(axis=1)

# Calculate the percentage of correct and incorrect guesses within each category
position_correctness_percentages = position_correctness_counts.div(totals, axis=0) * 100

# Plot the stacked bar chart
ax = position_correctness_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ff9999', '#66b3ff'], width=0.4)

# Customize the plot
plt.title('Correctness of Guesses by Batch Position')
plt.xlabel('Position Category')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)  # Rotate the x-axis labels to be horizontal
plt.legend(['Incorrect', 'Correct'], title='Guess')

# Initialize a list to keep track of the cumulative height of the bars
cumulative_heights = dict.fromkeys(position_correctness_counts.index, 0)

# Annotate the bars with percentages
for rect, (pos_cat, correct_guess), percentage in zip(ax.patches, position_correctness_percentages.stack().index, position_correctness_percentages.stack()):
    # Update the cumulative height for the current bar
    cumulative_heights[pos_cat] += rect.get_height()
    # Calculate the position for the label
    label_position = cumulative_heights[pos_cat] - (rect.get_height() / 2)
    # Place the label in the middle of the segment
    ax.text(rect.get_x() + rect.get_width() / 2, label_position, f'{percentage:.1f}%', ha='center', va='center')

plt.tight_layout()
plt.show()




