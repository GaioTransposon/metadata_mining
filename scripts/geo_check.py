#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:17:28 2024

@author: dgaio
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import glob
import os

# Set the working directory
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/production'

# Build the file pattern
file_pattern = os.path.join(work_dir, 'gpt_clean_output*.csv')

# List all CSV files that match the pattern
files = glob.glob(file_pattern)

# Initialize an empty DataFrame to hold all the data
df_geo_text = pd.DataFrame()

# Loop through each file and read the required columns
for file in files:
    df = pd.read_csv(file, usecols=['sample_id', 'geo_location'])
    df_geo_text = pd.concat([df_geo_text, df], ignore_index=True)

df_geo_text = df_geo_text.dropna(subset=['geo_location'])

# Display the cleaned DataFrame
print(df_geo_text)



data2 = {
    'sample_id': ['SRS3624032', 'SRS2358196', 'SRS5122277'],
    'latitude': [48.8575, 31.2304, 24.8797],
    'longitude': [2.3514, 121.4737, 102.8332]
}
df_coordinates = pd.DataFrame(data2)


# Merge on sample_id
df = pd.merge(df_geo_text, df_coordinates, on='sample_id')







# Set up geocoder
geolocator = Nominatim(user_agent="geoapiExercises")

# Function to geocode location
def geocode_location(location):
    try:
        loc = geolocator.geocode(location)
        return (loc.latitude, loc.longitude)
    except:
        return (None, None)

# Apply geocoding
df['geo_text_coords'] = df['geo_location'].apply(geocode_location)

# Calculate distances
def calculate_distance(row):
    if None not in row['geo_text_coords']:
        original_coords = (row['latitude'], row['longitude'])
        return geodesic(original_coords, row['geo_text_coords']).km
    else:
        return None

df['distance_km'] = df.apply(calculate_distance, axis=1)

# Define agreement based on distance
agreement_threshold = 10  # kilometers
df['agreement'] = df['distance_km'].apply(lambda x: x <= agreement_threshold if x is not None else None)

# Analyze results
agreement_rate = df['agreement'].mean() * 100
print(f"Agreement rate: {agreement_rate}%")
