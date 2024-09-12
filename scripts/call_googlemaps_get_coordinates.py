#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:40:51 2024

@author: dgaio
"""

import os
import pandas as pd
import requests

class GoogleMapsLocationCache:
    def __init__(self, work_dir, api_key_file, cache_filename):
        # Initialize paths
        self.work_dir = work_dir
        self.api_key_file = api_key_file
        self.geolocation_cache_file_path = os.path.join(self.work_dir, cache_filename)
        
        # Load or create the cache DataFrame
        if os.path.exists(self.geolocation_cache_file_path):
            self.cache_df = pd.read_csv(self.geolocation_cache_file_path)
        else:
            self.cache_df = pd.DataFrame(columns=['gpt_name', 'latitude', 'longitude'])

        # Load API key
        with open(self.api_key_file, 'r') as file:
            self.api_key = file.read().strip()

    def get_coordinates_from_cache(self, geo_location):
        """ Return coordinates from cache if available. """
        cached = self.cache_df[self.cache_df['gpt_name'] == geo_location]
        if not cached.empty:
            return cached.iloc[0]['latitude'], cached.iloc[0]['longitude']
        else:
            return None, None

    def fetch_coordinates(self, geo_location):
        """ Fetch coordinates from Google Maps API and update cache. """
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": geo_location,
            "key": self.api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            json_response = response.json()
            if json_response['results']:
                #print(json_response['results'])
                location = json_response['results'][0]['geometry']['location']
                # Add to cache
                new_row = pd.DataFrame([[geo_location, location['lat'], location['lng']]], 
                                       columns=['gpt_name', 'latitude', 'longitude'])
                self.cache_df = pd.concat([self.cache_df, new_row], ignore_index=True)
                return location['lat'], location['lng']
            else:
                return None, None
        else:
            return None, None

    def update_cache(self, geo_location_list):
        """ Iterate over the geo_location list and fetch coordinates if not in cache. """
        for geo in geo_location_list:
            lat, lon = self.get_coordinates_from_cache(geo)
            if lat is None and lon is None:  # Not in cache, fetch from API
                lat, lon = self.fetch_coordinates(geo)

        # Save updated cache to file
        self.cache_df.to_csv(self.geolocation_cache_file_path, index=False)
        return self.cache_df







# =============================================================================
# def fetch_country_info(country_name):
#     url = f"https://restcountries.com/v3.1/name/{country_name}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         return data[0]['area']  # area in square km
#     else:
#         return None
# 
# # Example usage
# area = fetch_country_info('Antarctica')
# print(area, 'square kilometers')
# =============================================================================

