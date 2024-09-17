#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:17:28 2024

@author: dgaio
"""


import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from call_googlemaps_get_coordinates import GoogleMapsLocationCache
from location_validation import LocationValidationGame
from math import radians, cos, sin, sqrt, atan2
import folium
import random
from collections import defaultdict, Counter
import json

# paths:
work_dir = os.path.join(os.path.expanduser('~'), 'cloudstor/Gaio/MicrobeAtlasProject')
middle_dir = os.path.join(os.path.expanduser('~'), 'github/metadata_mining/middle_dir')
out_dir = os.path.join(os.path.expanduser('~'), 'github/metadata_mining/out')
directory_with_split_metadata = 'sample.info_split_dirs'

# files
file_pattern = os.path.join(work_dir, 'production/gpt_clean_output*.csv')
coordinates_file = 'sample.coordinates.reparsed.filtered'
translated_coordinates = 'geocoded_coordinates.csv'
api_key_file = os.path.join(os.path.expanduser('~'), 'google_maps_api_key')
random_misclassified_samples_dict = 'random_misclassified_samples_dict.pkl'

# output files: 
map_all_matches = os.path.join(out_dir, 'map_with_color_coded_points_all.html')
map_all_mismatches = os.path.join(out_dir, 'map_with_color_coded_points_mismatches.html')


# 1. open gpt files and concatenate them: 
files = glob.glob(file_pattern)
gpt_geo_text = pd.concat(
    (pd.read_csv(f, usecols=['sample_id', 'geo_location']) for f in files),
    ignore_index=True
).dropna(subset=['geo_location'])

gpt_geo_text['geo_location'] = gpt_geo_text['geo_location'].str.replace(':', ' ', regex=False)
gpt_geo_text['geo_location'] = gpt_geo_text['geo_location'].str.replace('US', 'United States', regex=False)
gpt_geo_text['geo_location'] = gpt_geo_text['geo_location'].str.replace('USA', 'United States of America', regex=False)
gpt_geo_text['geo_location'] = gpt_geo_text['geo_location'].str.replace('Viet Nam', 'Vietnam', regex=False)
gpt_geo_text['geo_location'] = gpt_geo_text['geo_location'].str.replace('Czech Republic', 'Czechia', regex=False)
gpt_geo_text.rename(columns={'geo_location':'gpt_name'}, inplace=True)


# 2. open coordinates file
coordinates_file = os.path.join(middle_dir, coordinates_file)
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter=' ', header=None, names=['label', 'sample_id', 'latitude', 'longitude'], na_values='None')
df_coordinates_ori.drop(columns=['label'], inplace=True)  # Drop the label column as it's not needed


# 3. open translated coordinates file: 
translated_coordinates = os.path.join(middle_dir, translated_coordinates)
df_translated_coordinates = pd.read_csv(translated_coordinates)
df_translated_coordinates.rename(columns={'place_name':'latlon_name'}, inplace=True)


# merge 1.2.3.
merged_coordinates = pd.merge(df_coordinates_ori, df_translated_coordinates, on=['latitude', 'longitude'], how='left')
filtered_coordinates = merged_coordinates.dropna(subset=['latlon_name'])

# keep only samples present in gpt file
final_merge = pd.merge(filtered_coordinates, gpt_geo_text, on='sample_id', how='right')
# remove if coordinates not available
final_merge = final_merge.dropna(subset=['latitude'])



#######################################################


# Check if gpt locations match with Christian's extracted coordinates: 
def location_matches(place_name, geo_location):
    # Normalize strings
    place_name = place_name.lower()
    geo_location = geo_location.lower()

    # Extract country and city names (or any significant location identifier)
    keywords = set(re.split(r'[,\-\s:]\s*', place_name))

    # Check if any keyword is present in the geo_location
    return any(keyword in geo_location for keyword in keywords)

# Apply the comparison function
final_merge['location_match'] = final_merge.apply(lambda row: location_matches(row['latlon_name'], row['gpt_name']), axis=1)

# Get the counts of True vs. False matches
location_match_counts = final_merge['location_match'].value_counts()
total_samples = final_merge['location_match'].count()
true_matches_percentage = (location_match_counts.get(True, 0) / total_samples) * 100
false_matches_percentage = (location_match_counts.get(False, 0) / total_samples) * 100

# Print the results
print(f"Total Samples: {total_samples}")
print(f"Matches: {location_match_counts.get(True, 0)} ({true_matches_percentage:.2f}%)")
print(f"Non-matches: {location_match_counts.get(False, 0)} ({false_matches_percentage:.2f}%)")


#######################################################


# Use google maps API to ask coordinates corresponding to gpt_name: 

# Unique false matches including sample_id
false_matches = final_merge[final_merge['location_match'] == False]
columns_of_interest = ['sample_id', 'latitude', 'longitude', 'latlon_name', 'gpt_name', 'location_match']
unique_false_matches_counts = false_matches.groupby(columns_of_interest).size().reset_index(name='count')

# Update cache with unique gpt_name to avoid redundant API calls
unique_gpt_names = unique_false_matches_counts['gpt_name'].unique()

# call class to retrieve coordinates from geo_location (uses google maps api)
geo_cache = GoogleMapsLocationCache(work_dir, api_key_file, 'geolocation_cache.csv')
maps_coordinates = geo_cache.update_cache(unique_gpt_names)  
updated_coordinates = pd.read_csv(os.path.join(work_dir, 'geolocation_cache.csv'))
merged_false_matches = pd.merge(unique_false_matches_counts, updated_coordinates, on='gpt_name', how='left', suffixes=('_original', '_google'))


#######################################################


# Calculate distance between coordinate pairs: 

# Haversine formula for calculating the distance between two lat/lon pairs
def haversine(lat1, lon1, lat2, lon2):
    # Check if any of the inputs are NaN and return NaN if so
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
        return np.nan
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Apply Haversine formula to calculate distances
merged_false_matches['distance_km'] = merged_false_matches.apply(
    lambda row: haversine(row['latitude_original'], row['longitude_original'], row['latitude_google'], row['longitude_google']), axis=1)

# Drop rows where the distance is NaN due to missing coordinates
merged_false_matches = merged_false_matches.dropna(subset=['distance_km'])

# Calculate weighted distance by repeating the distance based on the 'count' column
weighted_distances = np.repeat(merged_false_matches['distance_km'], merged_false_matches['count'])

# # Plot histogram using the weighted distances
# plt.figure(figsize=(10, 6))
# plt.hist(weighted_distances, bins=60, color='blue', alpha=0.7)
# plt.title('Distance between Original and Google Coordinates (km) of misclassified samples')
# plt.xlabel('Distance (km)')
# plt.ylabel('Frequency (Weighted by Count)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# Compute summary statistics for the weighted distances
mean_distance = np.nanmean(weighted_distances)
median_distance = np.nanmedian(weighted_distances)
std_dev_distance = np.nanstd(weighted_distances)
min_distance = np.nanmin(weighted_distances)
max_distance = np.nanmax(weighted_distances)
percentile_25 = np.nanpercentile(weighted_distances, 25)
percentile_75 = np.nanpercentile(weighted_distances, 75)

# Print the summary statistics
print("Distance summary stats (km):")
print(f"Mean Distance: {mean_distance:.2f} km")
print(f"Median Distance: {median_distance:.2f} km")
print(f"Standard Deviation: {std_dev_distance:.2f} km")
print(f"Minimum Distance: {min_distance:.3f} km")
print(f"Maximum Distance: {max_distance:.2f} km")
print(f"25th Percentile: {percentile_25:.2f} km")
print(f"75th Percentile: {percentile_75:.2f} km")



#######################################################
###################### Visualization all ##############
#######################################################


# Create a base map at a central point
map = folium.Map(location=[final_merge['latitude'].mean(), final_merge['longitude'].mean()], zoom_start=5)

# Function to choose color based on match status
def get_color(match):
    return 'green' if match else 'red'

# Add points to the map
for idx, row in final_merge.iterrows():
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=3,
        color=get_color(row['location_match']),
        fill=True,
        fill_color=get_color(row['location_match']),
        fill_opacity=0.7,
        popup=f"Sample ID: {row['sample_id']}<br>Match: {row['location_match']}"
    ).add_to(map)

# Save the map as an HTML file
map.save(map_all_matches)

#######################################################
###################### Visualization mismatches #######
#######################################################

# drop 'sample_id' and get counts for mapping purposes:
mapped_data = merged_false_matches.drop(columns='sample_id').groupby(['latitude_original', 'longitude_original', 'latlon_name', 'gpt_name', 'distance_km']).size().reset_index(name='count')

map = folium.Map(location=[mapped_data['latitude_original'].mean(), mapped_data['longitude_original'].mean()], zoom_start=5)

def get_color(distance):
    if 0 < distance <= 100:
        return '#ADD8E6'  # Light Blue
    elif 100 < distance <= 500:
        return '#32CD32'  # Green
    elif 500 < distance <= 1000:
        return '#FFFF00'  # Yellow
    elif 1000 < distance <= 4000:
        return '#FFA500'  # Orange
    else:
        return '#FF0000'  # Red

# determine the radius based on count
def get_radius(count):
    if count == 1:
        return 2  # Tiniest dot
    elif 2 <= count <= 30:
        return 5  # Medium size
    else:
        return 8  # Large size

# Add points to the map
for idx, row in mapped_data.iterrows():
    folium.CircleMarker(
        location=(row['latitude_original'], row['longitude_original']),
        radius=get_radius(row['count']),
        color=get_color(row['distance_km']),
        fill=True,
        fill_color=get_color(row['distance_km']),
        fill_opacity=0.7,
        popup=(f"<div style='margin:10px;'><strong>Lat/Lon Name:</strong> {row['latlon_name']}<br>"
               f"<strong>GPT Name:</strong> {row['gpt_name']}<br>"
               f"<strong>Distance (km):</strong> {row['distance_km']:.2f}<br>"
               f"<strong>Count:</strong> {row['count']}</div>")
    ).add_to(map)

# Save the map as an HTML file
map.save(map_all_mismatches)

#######################################################
#######################################################



# Pick 200 random samples from the false matches (distance >100km), 
# unless this file had already been made (in case: load it). 

# Function to save the dictionary to a file
def save_samples_dict(samples_dict, filepath):
    pd.to_pickle(samples_dict, filepath)
    print(f"Sample dictionary saved to {filepath}")

# Function to load the dictionary from a file
def load_samples_dict(filepath):
    return pd.read_pickle(filepath)

# Check if the dictionary file exists
random_misclassified_samples_dict = os.path.join(middle_dir, random_misclassified_samples_dict)
if os.path.exists(random_misclassified_samples_dict):
    # Load the dictionary from the file
    random_samples_dict = load_samples_dict(random_misclassified_samples_dict)
    print("Loaded the sample dictionary from file.")
else:
    # Filter the DataFrame for entries with a distance greater than 1000 km
    high_distance_samples = merged_false_matches[merged_false_matches['distance_km'] > 1000]

    # List to keep track of unique (gpt_name, latlon_name) pairs
    unique_pairs = set()

    # Function to ensure values are not identical to another sample
    def is_unique_and_add(row):
        pair = (row['gpt_name'], row['latlon_name'])
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            return True
        return False

    # Filter for unique pairs
    unique_high_distance_samples = high_distance_samples[high_distance_samples.apply(is_unique_and_add, axis=1)]

    # Check if there are at least 200 samples
    if len(unique_high_distance_samples) >= 200:
        random_samples = unique_high_distance_samples.sample(n=200, random_state=1)  # Using a fixed seed for reproducibility
    else:
        # If fewer than 200 samples meet the criteria, take all available samples
        random_samples = unique_high_distance_samples
        print(f"Only {len(unique_high_distance_samples)} unique samples found with distance > 1000 km.")

    # Create a dictionary from the random samples with 'sample_id' as keys and ['gpt_name', 'latlon_name', 'distance_km'] as values
    random_samples_dict = random_samples.set_index('sample_id')[['gpt_name', 'latlon_name', 'distance_km']].to_dict('index')
    
    # Save the dictionary to a file
    save_samples_dict(random_samples_dict, random_misclassified_samples_dict)

# Print the number of samples in the dictionary
print(f"Number of samples in dictionary: {len(random_samples_dict)}")



#######################################################
#######################################################


# Play a game (it's a class outside of this script)
# what LocationValidationGame() does: 
# for each sample in dictionary, fetch its metadata and displays it, 
# shows gpt_name and latlon_name
# asks user is gpt (G), latlon (C), neither (N), or both (B) correct? 
# then prompt user to add a comment: " "
# Each time dictionary is updated with answer (G/C/N/B) as a value, and comment as another value of the dictionary. 
# For example: 
#     metadata for sample XXXXXX: 
#     ........
#     gpt location: .....
#     coordinates from metadata: .....
#     Who is right: 
#         gpt (G)
#         coordinates (C)
#         both (B)
#         neither (N)
#     Comment: 
#     {fill out expalaining why mistake happened} 

directory_with_split_metadata = os.path.join(work_dir, directory_with_split_metadata)
game = LocationValidationGame(random_samples_dict, directory_with_split_metadata, work_dir)
game.play()

# Get the updated data with user responses
updated_data = game.get_updated_data()
#print(updated_data)

# Count keys with more than 2 values
count = sum(1 for values in updated_data.values() if len(values) == 5)
print('Samples that have been manually validated: ',count) 


# =============================================================================
# # To edit comments once they are made: 
# # Replacements for comments under a certain category (e.g.: 'B')
# comment_replacements = {
#     "geo_loc_point_to_institute": "gpt_took_institute"
# }
# 
# # Process the data to replace comments only for entries under answer 'B'
# for key, value in updated_data.items():
#     if len(value) >= 5:  # Ensuring each entry has at least 5 key-value pairs
#         answer = value.get('answer', '')  # Safely getting the 'answer' field
#         if answer == 'C':  # Apply replacements only if the answer is 'B'
#             original_comment = value.get('comment', '').strip()
#             # Replace the comment if applicable
#             if original_comment in comment_replacements:
#                 value['comment'] = comment_replacements[original_comment]
#                 
# 
# # After modifications, save the updated data back to a JSON file
# output_file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/validation_game_progress.json'
# with open(output_file_path, 'w') as file:
#     json.dump(updated_data, file, indent=4)
# 
# print("Updated data has been saved to:", output_file_path)
# 
# =============================================================================






#######################################################
#######################################################


# Stats of misclassified samples (i.e.: gpt_name doesn't match with extracted coordinates from metadata): 

# Count each answer type and comments per answer
answer_counts = Counter()
comments_by_answer = defaultdict(Counter)
for details in updated_data.values():
    if 'answer' in details:
        answer = details['answer']
        comment = details.get('comment', 'No Comment')  # Use a default if no comment
        answer_counts[answer] += 1
        comments_by_answer[answer][comment] += 1

# Desired total samples
target_size = 100

# Calculate target counts for sampling
total_answers = sum(answer_counts.values())
targets = {answer: int(round((count / total_answers) * target_size)) for answer, count in answer_counts.items()}

# Adjust for rounding differences
adjustment = target_size - sum(targets.values())
targets[next(iter(targets))] += adjustment  # Adjust the first element

# Sample data to maintain proportional distribution
sampled_data = {}
for answer in targets:
    filtered = {k: v for k, v in updated_data.items() if v.get('answer') == answer}
    sampled = dict(random.sample(filtered.items(), min(len(filtered), targets[answer])))
    sampled_data.update(sampled)

# Output results
print(f"Total samples in subsampled data: {len(sampled_data)}")
print("Answer distribution in subsampled data:")
for answer, count in answer_counts.items():
    percentage = (count / total_answers * 100)
    print(f"{answer}: {count} ({percentage:.2f}%)")

print("\nDetailed comments distribution per answer category:")
for answer, comments in comments_by_answer.items():
    total_comments = sum(comments.values())
    print(f"\nAnswer {answer}:")
    for comment, count in comments.items():
        percentage = (count / total_comments * 100)
        print(f"{comment}: {count} ({percentage:.2f}%)")
