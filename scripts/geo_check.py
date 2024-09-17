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


# paths
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'
file_pattern = os.path.join(work_dir, 'production/gpt_clean_output*.csv')
coordinates_file = 'sample.coordinates.reparsed.filtered'
coordinates_file = os.path.join(work_dir, coordinates_file)
translated_coordinates = 'geocoded_coordinates.csv'
translated_coordinates = os.path.join(work_dir, translated_coordinates)
api_key_file = os.path.join(os.path.expanduser('~'), 'google_maps_api_key')
directory_with_split_metadata = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs'


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
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter=' ', header=None, names=['label', 'sample_id', 'latitude', 'longitude'], na_values='None')
df_coordinates_ori.drop(columns=['label'], inplace=True)  # Drop the label column as it's not needed


# 3. open translated coordinates file: 
df_translated_coordinates = pd.read_csv(translated_coordinates)
df_translated_coordinates.rename(columns={'place_name':'latlon_name'}, inplace=True)


# merge 1.2.3.
merged_coordinates = pd.merge(df_coordinates_ori, df_translated_coordinates, on=['latitude', 'longitude'], how='left')
filtered_coordinates = merged_coordinates.dropna(subset=['latlon_name'])

# keep only samples present in gpt file
final_merge = pd.merge(filtered_coordinates, gpt_geo_text, on='sample_id', how='right')
# remove if coordinates not available
final_merge = final_merge.dropna(subset=['latitude'])




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

# Visualize the count of True vs. False using a bar plot
location_match_counts = final_merge['location_match'].value_counts()
plt.figure(figsize=(8, 4))
location_match_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Proportion of Location Match True vs. False')
plt.xlabel('Location Match')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.6)  # Optional: adds grid lines for better readability
plt.show()


# 130689 samples





# Unique false matches including sample_id
false_matches = final_merge[final_merge['location_match'] == False]
columns_of_interest = ['sample_id', 'latitude', 'longitude', 'latlon_name', 'gpt_name', 'location_match']
unique_false_matches_counts = false_matches.groupby(columns_of_interest).size().reset_index(name='count')
print(unique_false_matches_counts)
print(f"Total number of unique false matches: {len(unique_false_matches_counts)}")

# Update cache with unique gpt_name to avoid redundant API calls
unique_gpt_names = unique_false_matches_counts['gpt_name'].unique()

# call class to retrieve coordinates from geo_location (uses google maps api)
geo_cache = GoogleMapsLocationCache(work_dir, api_key_file, 'geolocation_cache.csv')

maps_coordinates = geo_cache.update_cache(unique_gpt_names)  # This should update or write to 'geolocation_cache.csv'

updated_coordinates = pd.read_csv(os.path.join(work_dir, 'geolocation_cache.csv'))
merged_false_matches = pd.merge(unique_false_matches_counts, updated_coordinates, on='gpt_name', how='left', suffixes=('_original', '_google'))
print(merged_false_matches)






# Haversine formula for calculating the distance between two lat/lon pairs
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance



# Calculate the distance between the original lat/lon and the lat/lon from Google Maps
merged_false_matches['distance_km'] = merged_false_matches.apply(
    lambda row: haversine(row['latitude_original'], row['longitude_original'], row['latitude_google'], row['longitude_google']), axis=1)

# Display the result with distance
print(merged_false_matches[['gpt_name', 'latitude_original', 'longitude_original', 'latitude_google', 'longitude_google', 'distance_km']])

# Calculate weighted distance by repeating the distance based on the 'count' column
weighted_distances = np.repeat(merged_false_matches['distance_km'], merged_false_matches['count'])
len(weighted_distances)

# Plot histogram using the weighted distances
plt.figure(figsize=(10, 6))
plt.hist(weighted_distances, bins=60, color='blue', alpha=0.7)
plt.title('Distance between Original and Google Coordinates (km)')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency (Weighted by Count)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




# percentage of samples where gpt name matches with coordinates from metadata 
total_samples = len(final_merge)
matches_count = final_merge['location_match'].sum()
non_matches_count = total_samples - matches_count
matches_percentage = (matches_count / total_samples) * 100
non_matches_percentage = (non_matches_count / total_samples) * 100

print(f"Total samples: {total_samples}")
print(f"Matches: {matches_count} ({matches_percentage:.2f}%)")
print(f"Non-matches: {non_matches_count} ({non_matches_percentage:.2f}%)")



# how many false matches are sea/oceans/etc?
x = merged_false_matches[merged_false_matches['gpt_name'].str.contains('ocean|sea|lake', case=False, na=False)]
len(merged_false_matches)
len(x)




# how many samples per distance-category 
bins = [0, 100, 500, 1000, 4000, np.inf]
merged_false_matches['contains_keyword'] = merged_false_matches['gpt_name'].str.contains('ocean|sea|lake', case=False, na=False)
distance_category_counts = merged_false_matches['distance_category'].value_counts().sort_index()
total_samples = merged_false_matches.shape[0]
distance_category_percentages = (distance_category_counts / total_samples) * 100
keyword_counts = merged_false_matches[merged_false_matches['contains_keyword']].groupby('distance_category').size()
keyword_percentages = (keyword_counts / distance_category_counts) * 100
keyword_counts = keyword_counts.reindex(distance_category_counts.index, fill_value=0)
keyword_percentages = keyword_percentages.reindex(distance_category_counts.index, fill_value=0)
results = pd.DataFrame({
    'Total Count': distance_category_counts,
    'Total Percentage': distance_category_percentages,
    'Keyword Count': keyword_counts,
    'Keyword Percentage': keyword_percentages
})
print(results)





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
map.save('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/map_with_color_coded_points.html')

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
map.save('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/map_with_color_coded_points_false.html')



#######################################################
#######################################################



# Pick 200 samples from the false matches (distance >100km): 

# Filter the DataFrame for entries with a distance greater than 1000 km
high_distance_samples = merged_false_matches[merged_false_matches['distance_km'] > 1000]

# List to keep track of unique (gpt_name, latlon_name) pairs
unique_pairs = set()

# when picking random samples make sure the values are not identical to another sample.
def is_unique_and_add(row):
    pair = (row['gpt_name'], row['latlon_name'])
    if pair not in unique_pairs:
        unique_pairs.add(pair)
        return True
    return False

# Filter the high_distance_samples for unique pairs
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

# Print the dictionary to verify contents
print(random_samples_dict)
len(random_samples_dict)


# Make a game (as a class outside of this script)
# what game does: 
# for each sample in dictionary, fetch its metadata and display it, 
# show gpt_name and latlon_name
# ask user is gpt (G), latlon (C), neither (N), or both (B) correct? 
# then prompt user to add a comment: " "
# Each time dictionary is updated with answer (G/C/N/B) as a value, and comment as another value of the dictionary. 
# For example: 
#     metadata for sample XXXXXX: 
#     ........
#     gpt location: .....
#     coordinates from metadata: .....
#     Who is right: 
#         gpt
#         coordinates
#         both 
#         neither


game = LocationValidationGame(random_samples_dict, directory_with_split_metadata, work_dir)
game.play()

# Get the updated data with user responses
updated_data = game.get_updated_data()
print(updated_data)

# Count keys with more than 2 values
count = sum(1 for values in updated_data.values() if len(values) == 5)
print(count) 






# Extract the 5th value (assuming it's 'answer') from each entry that has 5 or more key-value pairs
answer_counts = {}
comments_by_answer = {}

for key, value in updated_data.items():
    if len(value) >= 5:  # Check if there are at least 5 key-value pairs
        answer = value['answer']  # Extract the 'answer' based on your data structure
        comment = value.get('comment', '').strip()  # Safely get the 'comment', strip whitespace, default to empty string

        # Count the occurrences of each answer
        if answer in answer_counts:
            answer_counts[answer] += 1
        else:
            answer_counts[answer] = 1

        # Collect comments by answer category and count them
        if answer not in comments_by_answer:
            comments_by_answer[answer] = {}
        if comment in comments_by_answer[answer]:
            comments_by_answer[answer][comment] += 1
        else:
            comments_by_answer[answer][comment] = 1

# Print answer counts
print("Answer Counts:", answer_counts)

# Print comments for each answer category with counts
for answer, comments in comments_by_answer.items():
    print(f"\nComments for answer {answer}:")
    for comment, count in comments.items():
        print(f"{count} {comment}")





# =============================================================================
# # Define the replacement rules for comments under the 'B' category
# comment_replacements = {
#     "cooridnates more precise": "coordinates_more_precise"
# }
# 
# # Process the data to replace comments only for entries under answer 'B'
# for key, value in updated_data.items():
#     if len(value) >= 5:  # Ensuring each entry has at least 5 key-value pairs
#         answer = value.get('answer', '')  # Safely getting the 'answer' field
#         if answer == 'B':  # Apply replacements only if the answer is 'B'
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
# =============================================================================


import random
from collections import defaultdict



# Extract counts of each answer type
answer_counts = defaultdict(int)
for details in updated_data.values():
    if 'answer' in details:  # Check if 'answer' key exists
        answer_counts[details['answer']] += 1
    else:
        # Handle cases with no 'answer' key; you might want to log these or handle them separately
        print(f"Missing 'answer' key for entry: {details}")

# Total to subsample to
target_size = 100

# Calculate proportions and target subsample sizes
total_answers = sum(answer_counts.values())
proportions = {answer: count / total_answers for answer, count in answer_counts.items()}
targets = {answer: int(round(proportions[answer] * target_size)) for answer in answer_counts}

# Adjust targets to exactly match target_size (due to rounding issues)
difference = target_size - sum(targets.values())
if difference != 0:
    # Adjust the category with the largest fraction leftover from rounding
    max_key = max(proportions, key=lambda k: proportions[k] - targets[k])
    targets[max_key] += difference

# Collect samples
sampled_data = {}
for answer, count in targets.items():
    # Filter entries by answer type and randomly pick the needed amount
    filtered = {k: v for k, v in updated_data.items() if v.get('answer') == answer}
    sampled = dict(random.sample(filtered.items(), count))
    sampled_data.update(sampled)

# Now `sampled_data` contains exactly 100 items with maintained proportions
print(f"Total samples in subsampled data: {len(sampled_data)}")
print("Answer distribution in subsampled data:")
for answer in ['B', 'G', 'C', 'N']:
    print(f"{answer}: {sum(1 for v in sampled_data.values() if v['answer'] == answer)}")

