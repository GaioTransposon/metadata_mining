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


# Unique false matches
false_matches = final_merge[final_merge['location_match'] == False]
columns_of_interest = ['latitude', 'longitude', 'latlon_name', 'gpt_name', 'location_match']
unique_false_matches_counts = false_matches.groupby(columns_of_interest).size().reset_index(name='count')
print(unique_false_matches_counts)
print(f"Total number of unique false matches: {len(unique_false_matches_counts)}")


# call class to retrieve coordinates from geo_location (uses google maps api)_
geo_cache = GoogleMapsLocationCache(work_dir, api_key_file, 'geolocation_cache.csv')

# update based on the False values 
maps_coordinates = geo_cache.update_cache(unique_false_matches_counts['gpt_name'].unique())

print('Number of unique FALSE gpt_names: ', len(unique_false_matches_counts['gpt_name'].unique()))


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



# Merge the new coordinates with unique_false_matches_counts based on geo_location
merged_false_matches = pd.merge(unique_false_matches_counts, maps_coordinates, on='gpt_name', how='left', suffixes=('_original', '_google'))


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





# Assuming merged_false_matches contains only the false matches and the distance_km column
# Filter to remove any rows without a calculated distance (if necessary)
merged_false_matches = merged_false_matches[merged_false_matches['distance_km'].notna()]

# Create a base map
map = folium.Map(location=[merged_false_matches['latitude_original'].mean(), 
                           merged_false_matches['longitude_original'].mean()], zoom_start=5)

# Function to choose color based on distance
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

# Add points to the map
for idx, row in merged_false_matches.iterrows():
    folium.CircleMarker(
        location=(row['latitude_original'], row['longitude_original']),
        radius=3,
        color=get_color(row['distance_km']),
        fill=True,
        fill_color=get_color(row['distance_km']),
        fill_opacity=0.7,
        popup=(f"<div style='margin:10px;'><strong>Lat/Lon Name:</strong> {row['latlon_name']}<br>"
               f"<strong>GPT Name:</strong> {row['gpt_name']}<br>"
               f"<strong>Distance (km):</strong> {row['distance_km']:.2f}</div>")
    ).add_to(map)

# Save the map as an HTML file
map.save('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/map_with_color_coded_points_false.html')










