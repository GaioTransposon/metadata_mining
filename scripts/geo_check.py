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
geolocator = Nominatim(user_agent="zzzz")
geocode_with_rate_limit = RateLimiter(geolocator.reverse, min_delay_seconds=3)  # Adjust min_delay_seconds as per usage policy

# Assuming df_coordinates_unique is already defined and contains 'latitude' and 'longitude'
df_coordinates_unique['place_name'] = df_coordinates_unique.apply(
    lambda row: reverse_geocode(row['latitude'], row['longitude']), axis=1
)

print(df_coordinates_unique[['latitude', 'longitude', 'place_name']])


# # for testing:
##geolocator.reverse(('31.110000', '121.380000'), exactly_one=True)







# APPEND FICTIVE INFO (MISMATCHES)

import pandas as pd

# Assuming df_coordinates_unique is already defined
# Create two fictive rows with mismatching geo_location and place_name
fictive_data = [
    {
        'sample_id': 'SRS9999999',
        'geo_location': 'Lima, Peru',
        'latitude': -12.0464,
        'longitude': -77.0428,
        'place_name': 'Moscow, Russia'  # Completely different location
    },
    {
        'sample_id': 'SRS8888888',
        'geo_location': 'Sydney, Australia',
        'latitude': -33.8688,
        'longitude': 151.2093,
        'place_name': 'New York, USA'  # Completely different location
    }
]

# Convert the fictive data into a DataFrame
fictive_df = pd.DataFrame(fictive_data)

# Append the fictive rows to the existing DataFrame
df_coordinates_unique = df_coordinates_unique.append(fictive_df, ignore_index=True)












import os
import openai
import json
import openai
import numpy as np
import pandas as pd

# Set API key path and initialize OpenAI API
api_key_path = '/Users/dgaio/my_api_key_embeddings'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()

# Function to get embeddings from OpenAI API
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error retrieving embedding for text '{text}': {e}")
        return None

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Function to compare geo_location and place_name using embeddings
def compare_geo_location_with_place_name(row):
    geo_loc_embedding = get_embedding(row['geo_location'])
    place_name_embedding = get_embedding(row['place_name'])
    
    if geo_loc_embedding is not None and place_name_embedding is not None:
        similarity_score = cosine_similarity(geo_loc_embedding, place_name_embedding)
        return similarity_score >= 0.5, similarity_score
    else:
        return False, None


df = df_coordinates_unique

# Apply the comparison function to each row
df[['is_similar', 'similarity_score']] = df.apply(compare_geo_location_with_place_name, axis=1, result_type='expand')

# Print the results
print(df[['sample_id', 'geo_location', 'place_name', 'is_similar', 'similarity_score']])









import os
import openai
import numpy as np
import pandas as pd

# Set API key path and initialize OpenAI API
api_key_path = '/Users/dgaio/my_api_key_embeddings'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()

# Function to get embeddings from OpenAI API with juxtaposition
def get_embedding(text):
    try:
        # Adding a juxtaposition context to the input text
        response = openai.Embedding.create(input=[f"A dog in {text}"], engine="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error retrieving embedding for text '{text}': {e}")
        return None

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Function to compare geo_location and place_name using embeddings with juxtaposition
def compare_geo_location_with_place_name(row):
    # Adding the context "A dog in" to both geo_location and place_name
    geo_loc_embedding = get_embedding(f"A dog in {row['geo_location']}")
    place_name_embedding = get_embedding(f"A dog in {row['place_name']}")
    
    if geo_loc_embedding is not None and place_name_embedding is not None:
        similarity_score = cosine_similarity(geo_loc_embedding, place_name_embedding)
        return similarity_score >= 0.5, similarity_score
    else:
        return False, None

# Assuming df_coordinates_unique is your DataFrame
df = df_coordinates_unique

# Apply the comparison function to each row
df[['is_similar', 'similarity_score']] = df.apply(compare_geo_location_with_place_name, axis=1, result_type='expand')

# Print the results
print(df[['sample_id', 'geo_location', 'place_name', 'is_similar', 'similarity_score']])




