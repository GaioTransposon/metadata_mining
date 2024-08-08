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
import googlemaps

# Set the working directory for geolocation text files
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/production'
file_pattern = os.path.join(work_dir, 'gpt_clean_output*.csv')
files = glob.glob(file_pattern)

# Initialize an empty DataFrame to hold all the data from geolocation text files
df_geo_text = pd.DataFrame()

# Loop through each file and read the required columns
for file in files:
    df = pd.read_csv(file, usecols=['sample_id', 'geo_location'])
    df_geo_text = pd.concat([df_geo_text, df], ignore_index=True)

# Filter out rows where 'geo_location' is NaN
df_geo_text = df_geo_text.dropna(subset=['geo_location'])

# Path to the file with coordinates
coordinates_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/samples.info.latlon.parsed'

# Read the coordinates file, skipping the initial comment lines
df_coordinates_ori = pd.read_csv(coordinates_file, delimiter='\t', header=0)

print(df_coordinates_ori.columns)

df_coordinates_ori = df_coordinates_ori.rename(columns={'#sampleId': 'sample_id', 'parsedLat': 'latitude', 'parsedLon': 'longitude'})


len(df_coordinates_ori)

df_coordinates_ori.replace('None', np.nan, inplace=True)
df_coordinates_ori.dropna(subset=['latitude', 'longitude'], inplace=True)

len(df_coordinates_ori)


df_coordinates = df_coordinates_ori.copy()

df_coordinates['latitude'] = pd.to_numeric(df_coordinates['latitude'], errors='coerce')
df_coordinates['longitude'] = pd.to_numeric(df_coordinates['longitude'], errors='coerce')


# Merge on sample_id
df = pd.merge(df_geo_text, df_coordinates, on='sample_id')

len(df_geo_text)

len(df)
print(df)

df = df.drop(columns=['latFieldLabel', 'latFieldOriginalValue', 'lonFieldLabel', 'lonFieldOriginalValue'])







# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key='AIzaSyAstMiG7yR1Sm98NyFmSjZqCEFj1nnIrl0')

# Function to geocode location using Google Maps
def geocode_location(location):
    try:
        geocode_result = gmaps.geocode(location)
        # Extract latitude and longitude
        if geocode_result and len(geocode_result) > 0:
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']
            return (lat, lng)
        else:
            return (None, None)
    except Exception as e:
        print(f"Error geocoding {location}: {e}")
        return (None, None)





df = df.head(10)  
df[['text_latitude', 'text_longitude']] = df['geo_location'].apply(geocode_location).apply(pd.Series)



df_test = df.copy()




import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers (Approximately 6371)
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Adding a new column to the DataFrame to store distances
df_test['distance_km'] = df_test.apply(lambda row: haversine(row['latitude'], row['longitude'], row['text_latitude'], row['text_longitude']), axis=1)

# Display the DataFrame to check distances
print(df_test[['sample_id', 'geo_location', 'latitude', 'longitude', 'text_latitude', 'text_longitude', 'distance_km']])




def determine_threshold(geo_location):
    # Heuristic: if the location is likely a country (e.g., no specific city or region mentioned)
    if len(geo_location.split(':')) == 1:  # Simple check; might need more sophisticated checks
        return 1000  # Larger threshold for country-level
    else:
        return 50  # Smaller threshold for city-level

# Apply a variable threshold based on the specificity of the location
df_test['threshold'] = df_test['geo_location'].apply(determine_threshold)
df_test['agreement'] = df_test.apply(lambda row: row['distance_km'] < row['threshold'], axis=1)




df_test












import geopandas as gpd
import matplotlib.pyplot as plt

# Load the Shapefile
gdf = gpd.read_file('/Users/dgaio/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# Display the first few rows of the GeoDataFrame
print(gdf.head())



# Plotting points
fig, ax = plt.subplots()
gdf.plot(ax=ax, kind='scatter', x='LABEL_X', y='LABEL_Y', color='blue', label='Geo Labels')
plt.show()




# Calculate the centroid of each country (this gives you the central latitude and longitude)
gdf['centroid'] = gdf.geometry.centroid

# Calculate the bounding box of each country
gdf['bounds'] = gdf.geometry.bounds

# Extracting specific columns to display country, centroid and bounding boxes
gdf = gdf[['NAME', 'centroid', 'bounds']]

# Printing the results
print(gdf)




import matplotlib.pyplot as plt

# Plot the world map using the boundary data
gdf.plot()
plt.show()




import geopandas as gpd


# Path to your File Geodatabase
gdb_path = '/Users/dgaio/Downloads/gadm_410.gdb/a00000007.gdbtable'

# List all feature classes in a File Geodatabase
layers = fiona.listlayers(gdb_path)

for layer in layers:
    gdf = gpd.read_file(gdb_path, layer=layer)
    print(gdf.head())  # Print the first few rows of the GeoDataFrame






