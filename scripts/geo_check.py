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
import requests
import sys
sys.path.append('/Users/dgaio/github/metadata_mining/scripts')
from call_googlemaps_get_coordinates import GoogleMapsLocationCache
from math import radians, cos, sin, sqrt, atan2

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


# 2. open coordinates file
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter=' ', header=None, names=['label', 'sample_id', 'latitude', 'longitude'], na_values='None')
df_coordinates_ori.drop(columns=['label'], inplace=True)  # Drop the label column as it's not needed


# 3. open translated coordinates file: 
df_translated_coordinates = pd.read_csv(translated_coordinates)


# merge 1.2.3.
merged_coordinates = pd.merge(df_coordinates_ori, df_translated_coordinates, on=['latitude', 'longitude'], how='left')
filtered_coordinates = merged_coordinates.dropna(subset=['place_name'])

# keep only samples present in gpt file
final_merge = pd.merge(filtered_coordinates, gpt_geo_text, on='sample_id', how='right')
# remove if coordinates not available
final_merge = final_merge.dropna(subset=['latitude'])





# Check if gpt locations match with Chjristian's extracted coordinates: 
df = final_merge

df['geo_location'] = df['geo_location'].str.replace(':', ' ', regex=False)
df['geo_location'] = df['geo_location'].str.replace('US', 'United States', regex=False)
df['geo_location'] = df['geo_location'].str.replace('USA', 'United States of America', regex=False)
df['geo_location'] = df['geo_location'].str.replace('Viet Nam', 'Vietnam', regex=False)



def location_matches(place_name, geo_location):
    # Normalize strings
    place_name = place_name.lower()
    geo_location = geo_location.lower()

    # Extract country and city names (or any significant location identifier)
    keywords = set(re.split(r'[,\-\s:]\s*', place_name))


    # Check if any keyword is present in the geo_location
    return any(keyword in geo_location for keyword in keywords)

# Apply the comparison function
df['location_match'] = df.apply(lambda row: location_matches(row['place_name'], row['geo_location']), axis=1)

print(df[['sample_id', 'geo_location', 'place_name', 'location_match']])

df['location_match'].value_counts().get(False, 0)


# Visualize the count of True vs. False using a bar plot
location_match_counts = df['location_match'].value_counts()
plt.figure(figsize=(8, 4))
location_match_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Proportion of Location Match True vs. False')
plt.xlabel('Location Match')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.6)  # Optional: adds grid lines for better readability
plt.show()


# Unique false matches
false_matches = df[df['location_match'] == False]
columns_of_interest = ['latitude', 'longitude', 'place_name', 'geo_location', 'location_match']
unique_false_matches_counts = false_matches.groupby(columns_of_interest).size().reset_index(name='count')
print(unique_false_matches_counts)
print(f"Total number of unique false matches: {len(unique_false_matches_counts)}")

# call class to retrieve coordinates from geo_location (uses google maps api)_
geo_cache = GoogleMapsLocationCache(work_dir, api_key_file)

# update based on the False values 
maps_coordinates = geo_cache.update_cache(unique_false_matches_counts['geo_location'].unique())





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
merged_false_matches = pd.merge(unique_false_matches_counts, maps_coordinates, on='geo_location', how='left', suffixes=('_original', '_google'))

# Calculate the distance between the original lat/lon and the lat/lon from Google Maps
merged_false_matches['distance_km'] = merged_false_matches.apply(
    lambda row: haversine(row['latitude_original'], row['longitude_original'], row['latitude_google'], row['longitude_google']), axis=1)

# Display the result with distance
print(merged_false_matches[['geo_location', 'latitude_original', 'longitude_original', 'latitude_google', 'longitude_google', 'distance_km']])

# Visualize the distance between original and Google coordinates
plt.figure(figsize=(10, 6))
plt.hist(merged_false_matches['distance_km'], bins=30, color='blue', alpha=0.7)
plt.title('Distance between extracted coordinates and coordinates from gpt-retrieved-location (km)')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




import matplotlib.pyplot as plt
import numpy as np

# Calculate weighted distance by repeating the distance based on the 'count' column
weighted_distances = np.repeat(merged_false_matches['distance_km'], merged_false_matches['count'])

# Plot histogram using the weighted distances
plt.figure(figsize=(10, 6))
plt.hist(weighted_distances, bins=30, color='blue', alpha=0.7)
plt.title('Distance between Original and Google Coordinates (km)')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency (Weighted by Count)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


