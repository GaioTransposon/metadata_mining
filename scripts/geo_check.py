#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:17:28 2024

@author: dgaio
"""




import pandas as pd
import glob
import os
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# paths
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/production'
file_pattern = os.path.join(work_dir, 'gpt_clean_output*.csv')
files = glob.glob(file_pattern)
coordinates_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/samples.info.latlon.parsed'

# 1. Concatenate all gpt files
gpt_geo_text = pd.concat(
    (pd.read_csv(f, usecols=['sample_id', 'geo_location']) for f in files),
    ignore_index=True
).dropna(subset=['geo_location'])



# 2. read and filter coordinates file
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter='\t', header=0, na_values='None').rename(
    columns={'#sampleId': 'sample_id', 'parsedLat': 'latitude', 'parsedLon': 'longitude'}
).dropna(subset=['latitude', 'longitude'])

df_coordinates_ori['latitude'] = pd.to_numeric(df_coordinates_ori['latitude'], errors='coerce')
df_coordinates_ori['longitude'] = pd.to_numeric(df_coordinates_ori['longitude'], errors='coerce')
df_coordinates_ori = df_coordinates_ori[['sample_id', 'latitude', 'longitude']]



# Filter out invalid latitude or longitude values
df_coordinates_filtered = df_coordinates_ori[
    df_coordinates_ori['latitude'].between(-90, 90) & 
    df_coordinates_ori['longitude'].between(-180, 180)
].dropna(subset=['latitude', 'longitude'])

print('Samples with extracted coordinates: ', len(df_coordinates_filtered))


# Filter to keep only samples present in gpt_geo_text
df_coordinates_filtered = pd.merge(gpt_geo_text, df_coordinates_ori, on='sample_id')


# Remove duplicate coordinates, independent of sample_id
df_coordinates_unique = df_coordinates_filtered.drop_duplicates(subset=['latitude', 'longitude'])

print('Samples with extracted coordinates: ', len(df_coordinates_unique))



# Remove duplicate coordinates, independent of sample_id
df_coordinates_unique = df_coordinates_filtered.drop_duplicates(subset=['latitude', 'longitude'])

print('Unique samples with extracted coordinates: ', len(df_coordinates_unique))




df_coordinates_unique = df_coordinates_unique[1:10]





import pandas as pd

# Function to perform reverse geocoding
def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        if location:
            return location.address
        else:
            return None
    except:
        return None

# Initialize the Nominatim geocoder with RateLimiter to avoid hitting API usage limits
geolocator = Nominatim(user_agent="myexercises")
geocode_with_rate_limit = RateLimiter(geolocator.reverse, min_delay_seconds=1)  # Adjust min_delay_seconds as per usage policy

# Assuming df_coordinates_unique is already defined and contains 'latitude' and 'longitude'
df_coordinates_unique['place_name'] = df_coordinates_unique.apply(
    lambda row: reverse_geocode(row['latitude'], row['longitude']), axis=1
)

print(df_coordinates_unique[['latitude', 'longitude', 'place_name']])








