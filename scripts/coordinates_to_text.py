#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:17:21 2024

@author: dgaio
"""



import pandas as pd
import glob
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import argparse
import logging

# Set up argument parsing
parser = argparse.ArgumentParser(description='Read coordinates and translate to location names.')
parser.add_argument('--work_dir', type=str, required=True, help='work directory')
parser.add_argument('--coordinates_file', type=str, required=True, help='Christian s coordinates file')
parser.add_argument('--output_file', type=str, required=True, help='output filename (to save the translated cooridnates)')
parser.add_argument('--min_delay_seconds', type=float, default=1.5, help='Minimum delay for the geopy rate limiter (seconds)')
args = parser.parse_args()


# paths 
work_dir = args.work_dir
coordinates_file = os.path.join(args.work_dir, args.coordinates_file)
output_file = os.path.join(args.work_dir, args.output_file)


# read and filter coordinates file
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter=' ', header=None, names=['label', 'sample_id', 'latitude', 'longitude'], na_values='None')
df_coordinates_ori.drop(columns='label', inplace=True)  # Drop the 'label' column if it's not needed
df_coordinates_ori['latitude'] = pd.to_numeric(df_coordinates_ori['latitude'], errors='coerce')
df_coordinates_ori['longitude'] = pd.to_numeric(df_coordinates_ori['longitude'], errors='coerce')
df_coordinates_ori.dropna(subset=['latitude', 'longitude'], inplace=True)

# filter out invalid latitude or longitude values
df_coordinates_filtered = df_coordinates_ori[
    df_coordinates_ori['latitude'].between(-90, 90) & 
    df_coordinates_ori['longitude'].between(-180, 180)
].dropna(subset=['latitude', 'longitude'])


# drop column
df_coordinates_filtered = df_coordinates_filtered.drop(columns='sample_id')

# keep unique coordinates
df_coordinates_filtered = df_coordinates_filtered.drop_duplicates(subset=['latitude', 'longitude'])

df_coordinates_unique = df_coordinates_filtered

# Function to perform reverse geocoding
def reverse_geocode(lat, lon):
    if not results_df[(results_df['latitude'] == lat) & (results_df['longitude'] == lon)].empty:
        return results_df[(results_df['latitude'] == lat) & (results_df['longitude'] == lon)]['place_name'].iloc[0]
    try:
        location = geocode_with_rate_limit((lat, lon), exactly_one=True, language='en')
        if location:
            return location.address
        else:
            return None
    except Exception as e:
        logging.error(f"Error during geocoding for coordinates ({lat}, {lon}): {e}")
        return None

# Initialize Nominatim geocoder with RateLimiter
geolocator = Nominatim(user_agent="Microbe Atlas metadata project - coordinates translation")
geocode_with_rate_limit = RateLimiter(geolocator.reverse, min_delay_seconds=args.min_delay_seconds)

# Check if the results file already exists, load it if it does, else create an empty DataFrame
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
else:
    results_df = pd.DataFrame(columns=['latitude', 'longitude', 'place_name'])

# Loop to find location name for respective coordinates (saves after each hit)
n = 0
for index, row in df_coordinates_unique.iterrows():
    if not ((results_df['latitude'] == row['latitude']) & (results_df['longitude'] == row['longitude'])).any():
        n += 1
        print('Reverse geocoding...', n)
        place_name = reverse_geocode(row['latitude'], row['longitude'])
        new_row = pd.DataFrame([{'latitude': row['latitude'], 'longitude': row['longitude'], 'place_name': place_name}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(output_file, index=False)

print(results_df[['latitude', 'longitude', 'place_name']])



# =============================================================================
# python /Users/dgaio/github/metadata_mining/scripts/coordinates_to_text.py \
#     --work_dir "/mnt/mnemo5/dgaio/MicrobeAtlasProject" \
#     --coordinates_file "sample.coordinates.reparsed.filtered" \
#     --output_file "geocoded_coordinates.csv" \
#     --min_delay_seconds 1.5
# =============================================================================











