#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:17:28 2024

@author: dgaio
"""



import pandas as pd
import glob
import os
import numpy as np
#import googlemaps
import json
import random

# paths
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/production'
file_pattern = os.path.join(work_dir, 'gpt_clean_output*.csv')
files = glob.glob(file_pattern)
coordinates_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/samples.info.latlon.parsed'


## 1. 
# loop through each gpt file
df_geo_text = pd.DataFrame()
for file in files:
    df = pd.read_csv(file, usecols=['sample_id', 'geo_location'])
    df_geo_text = pd.concat([df_geo_text, df], ignore_index=True)

df_geo_text = df_geo_text.dropna(subset=['geo_location'])


df_geo_text['geo_location'] = df_geo_text['geo_location'].replace(
    to_replace=r'\bUSA\b', value='United States of America', regex=True)

df_geo_text['geo_location'] = df_geo_text['geo_location'].replace(
    to_replace=r'\bUnited States\b', value='United States of America', regex=True)

df_geo_text['geo_location'] = df_geo_text['geo_location'].replace(
    to_replace=r'\bUK\b', value='U.K. of Great Britain and Northern Ireland', regex=True)

df_geo_text['geo_location'] = df_geo_text['geo_location'].replace(
    to_replace=r'\bU.K.\b', value='U.K. of Great Britain and Northern Ireland', regex=True)



print('Samples with gpt text locations: ', len(df_geo_text))
####


## 2. 
# read and filter coordinates file
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter='\t', header=0)

df_coordinates_ori = df_coordinates_ori.rename(columns={'#sampleId': 'sample_id', 'parsedLat': 'latitude', 'parsedLon': 'longitude'})
df_coordinates_ori.replace('None', np.nan, inplace=True)
df_coordinates_ori.dropna(subset=['latitude', 'longitude'], inplace=True)

df_coordinates_ori['latitude'] = pd.to_numeric(df_coordinates_ori['latitude'], errors='coerce')
df_coordinates_ori['longitude'] = pd.to_numeric(df_coordinates_ori['longitude'], errors='coerce')


len(df_coordinates_ori)
df_coordinates_ori.columns

df_coordinates_ori_filtered = df_coordinates_ori[['sample_id', 'latitude', 'longitude']]



# Filter out invalid latitude or longitude values
df_coordinates_ori_filtered = df_coordinates_ori_filtered[
    (df_coordinates_ori['latitude'].between(-90, 90)) &
    (df_coordinates_ori['longitude'].between(-180, 180)) &
    df_coordinates_ori['latitude'].notna() &
    df_coordinates_ori['longitude'].notna()
]

print('Samples with extracted coordinates: ', len(df_coordinates_ori_filtered))



####

## 3. 
# merge on sample id keep unique (new column: samples_count)

df = pd.merge(df_geo_text, df_coordinates_ori_filtered, on='sample_id')

len(df_geo_text)
len(df_coordinates_ori_filtered)
len(df)

# Group by 'geo_location' and count 'sample_id', keeping all latitude and longitude entries
df = df.groupby(['geo_location', 'latitude', 'longitude']).agg({
    'sample_id': 'count'
}).rename(columns={'sample_id': 'sample_count'}).reset_index()

len(df)




## 4. 
# Datasets: 

# Load data from each CSV file
df_countries = pd.read_csv('/Users/dgaio/Downloads/world-administrative-boundaries.csv', delimiter=';')

df_states = pd.read_csv('/Users/dgaio/Downloads/us-state-boundaries.csv', delimiter=';')

#df_cities = pd.read_csv('/Users/dgaio/Downloads/geonames-all-cities-with-a-population-1000.csv', delimiter=';')


# Select and rename columns for the countries dataframe
df_countries = df_countries[['English Name', 'Geo Shape']].rename(columns={'English Name': 'name', 'Geo Shape': 'coordinates'})
df_countries['origin'] = 'countries'

# Select and rename columns for the states dataframe
df_states = df_states[['name', 'St Asgeojson']].rename(columns={'St Asgeojson': 'coordinates'})
df_states['origin'] = 'US_states'

#
# df_counties: 
# 	name	coordinates	origin
# 0	Uganda	"{""coordinates"": [[[33.92110000000008, -1.0019399999999337], [34.45389000000006, 3.2455600000000686], [33.92110000000008, -1.0019399999999337]]], ""type"": ""Polygon""}"	countries

# df_states:
# 	name	coordinates	origin
# 0	Vermont	"{""coordinates"": [[[-73.3132829996123, 44.2641300000495],  [-73.3132829996123, 44.2641300000495]]], ""type"": ""Polygon""}"	US_states

# df_cities:
# 	name	coordinates	origin
# 0	Heist-op-den-Berg	51.07537, 4.72827	cities


# Function to extract coordinates and randomly sample them, plus reverse lat lon
def extract_sample_and_reverse_coordinates(geojson_str):
    try:
        # Decode the JSON formatted string into a Python dictionary
        geojson = json.loads(geojson_str)
        all_coords = []

        # Check and process the coordinates based on geometry type
        if geojson['type'] == 'Polygon':
            # Get the first list of coordinates assuming there are no holes (or just the outer boundary)
            coordinates = geojson['coordinates'][0]
            sampled_coords = random.sample(coordinates, min(20, len(coordinates)))
            all_coords.extend([[lon, lat] for lat, lon in sampled_coords])
        elif geojson['type'] == 'MultiPolygon':
            # Flatten and sample coordinates from multiple polygons
            coordinates = [coord for poly in geojson['coordinates'] for coord in poly[0]]
            sampled_coords = random.sample(coordinates, min(20, len(coordinates)))
            all_coords.extend([[lon, lat] for lat, lon in sampled_coords])

        return all_coords
    except json.JSONDecodeError:
        print("Error decoding JSON from string:", geojson_str)
        return []
    except Exception as e:
        print("An error occurred:", str(e))
        return []


# def process_city_coordinates(coord_str):
#     # Split the string by comma to separate latitude and longitude
#     lat, lon = map(float, coord_str.split(','))
#     # Reverse the coordinates and wrap in a list of lists to match the other DataFrame formats
#     return [[lon, lat]]

random.seed(42)
df_countries['coordinates'] = df_countries['coordinates'].apply(extract_sample_and_reverse_coordinates)
df_states['coordinates'] = df_states['coordinates'].apply(extract_sample_and_reverse_coordinates)
#df_cities['coordinates'] = df_cities['coordinates'].apply(process_city_coordinates)

df_countries.columns
df_states.columns
#df_cities.columns

######



## 5. 




def prepare_dataframes(df):
    # Normalize and remove duplicates to prepare for matching
    df['normalized_name'] = df['name'].str.lower().str.strip()
    return df.drop_duplicates(subset='normalized_name')

# Prepare the dataframes
#df_cities = prepare_dataframes(df_cities)
df_states = prepare_dataframes(df_states)
df_countries = prepare_dataframes(df_countries)



# Cache for results to avoid recomputation
cache = {}

import re

def find_match(location):
    if location in cache:
        return cache[location]

    normalized_location = location.lower().strip()
    potential_matches = []

    # Changed order of checking: states, countries, and then cities
    for df, label in [(df_states, 'state'), (df_countries, 'country')]:
        matches = df[df['normalized_name'].apply(lambda x: x in normalized_location)]
        if not matches.empty:
            for _, match in matches.iterrows():
                # Check for full string match or structured inclusion
                if re.search(r'\b' + re.escape(match['normalized_name']) + r'\b', normalized_location):
                    # Assign scores, higher is better based on new prioritization
                    score = 10 if label == 'state' else (100 if label == 'country' else 1)
                    potential_matches.append((score, match['name'], match['coordinates'], label))

    if potential_matches:
        # Choose the match with the highest score
        best_match = max(potential_matches, key=lambda x: x[0])
        cache[location] = best_match[1:]
        return best_match[1:]

    cache[location] = (None, None, None)
    return (None, None, None)


# Applying the matching logic
df['best_match_name'], df['matched_coordinates'], df['match_type'] = zip(
    *df['geo_location'].map(find_match))

# Filter out NaN
df = df.dropna(subset=['best_match_name'])
print(df)



# =============================================================================
# # Collect non-matches
# non_matches = df_geo_text[df_geo_text['best_match_name'].isnull()]['geo_location'].value_counts()
# 
# print("Non-matches:", non_matches)
# len(non_matches)
# 
# # possibly find coordinates for geo location (text) using google maps: 
# gmaps = googlemaps.Client(key='AIzaSyAstMiG7yR1Sm98NyFmSjZqCEFj1nnIrl0')
# 
# gmaps.geocode('North Pacific Ocean Gyre')
# gmaps.geocode('Chicago')
# =============================================================================






# filter for samples present in the gpt files
df_coordinates_ori_filtered = df_coordinates_ori_filtered[df_coordinates_ori_filtered['sample_id'].isin(df_geo_text_filtered['sample_id'])]
len(df_coordinates_ori_filtered)



# subsampling for testing
t = df[1:1000]






import pandas as pd
from geopy.distance import geodesic

# Your function defined here...
def find_closest_point_and_distance(lat, lon, coordinates_list):
    if not coordinates_list:
        return (None, None)  # No coordinates to compare against

    min_distance = float('inf')
    closest_point = None
    n = 0

    for coords in coordinates_list:
        n += 1
        if len(coords) == 2:
            current_lat, current_lon = coords
            current_distance = geodesic((lat, lon), (current_lat, current_lon)).kilometers
            if current_distance < min_distance:
                min_distance = current_distance
                closest_point = coords
                print(n)  # Printing the loop iteration

    return closest_point, min_distance





# Assuming 'df' is your final DataFrame after merging and matching
t[['closest_point', 'min_distance']] = t.apply(
    lambda row: find_closest_point_and_distance(row['latitude'], row['longitude'], row['matched_coordinates']),
    axis=1, result_type='expand'
)





def get_rows_by_coordinates(df, lat, lon):
    filtered_df = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
    return filtered_df

get_rows_by_coordinates(df_coordinates_ori_filtered, 17.0333, -63.4167) # metadata coordinates wrong  
get_rows_by_coordinates(df_coordinates_ori_filtered, -35.2087, 26.2932) # gpt allucinating


df_geo_text[(df_geo_text['sample_id']=='ERS490053')]


