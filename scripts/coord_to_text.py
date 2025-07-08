#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:17:21 2024

@author: dgaio
"""


# run as: 

# python ~/github/metadata_mining/scripts/coord_to_text.py \
#     --work_dir ~/MicrobeAtlasProject \
#     --coordinates_file sample.coordinates.reparsed.filtered \
#     --output_file geocoded_coordinates.csv \
#     --min_delay_seconds 1.3

import pandas as pd
import glob
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import argparse
import logging
from geopy.exc import GeocoderTimedOut
import sys
import time
from datetime import datetime

# Set up argument parsing
parser = argparse.ArgumentParser(description='Read coordinates and translate to location names.')
parser.add_argument('--work_dir', type=str, required=True, help='work directory')
parser.add_argument('--coordinates_file', type=str, required=True, help='Christian''s coordinates file')
parser.add_argument('--output_file', type=str, required=True, help='output filename (to save the translated coordinates)')
parser.add_argument('--min_delay_seconds', type=float, default=1.5, help='Minimum delay for the geopy rate limiter (seconds)')
args = parser.parse_args()

logging.basicConfig(level=logging.ERROR, handlers=[logging.StreamHandler(sys.stderr)])

# Custom progress logging function
def log_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', log_file_path=None):
    """
    Log progress to both terminal and file
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    progress_msg = f'{prefix} |{bar}| {percent}% {suffix}'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Try multiple output methods
    try:
        # Method 1: Direct print to stdout/stderr
        print(f'\r{progress_msg}', end='\r')
        sys.stdout.flush()
        print(f'\r{progress_msg}', end='\r', file=sys.stderr)
        sys.stderr.flush()
        
        # Method 2: Write to log file
        if log_file_path:
            with open(log_file_path, 'a') as f:
                f.write(f'[{timestamp}] {progress_msg}\n')
                f.flush()
        
        # Method 3: Force output with os.write
        import os
        os.write(1, f'\r{progress_msg}\n'.encode())  # stdout
        os.write(2, f'\r{progress_msg}\n'.encode())  # stderr
        
    except Exception as e:
        # Fallback: just write to file
        if log_file_path:
            with open(log_file_path, 'a') as f:
                f.write(f'[{timestamp}] ERROR in progress display: {e}\n')
                f.write(f'[{timestamp}] {progress_msg}\n')
                f.flush()

# paths 
work_dir = args.work_dir
coordinates_file = os.path.join(args.work_dir, args.coordinates_file)
output_file = os.path.join(args.work_dir, args.output_file)
log_file = os.path.join(args.work_dir, 'geocoding_progress.log')

# Initialize log file
with open(log_file, 'w') as f:
    f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting geocoding process\n')
    f.flush()

# read and filter coordinates file
with open(log_file, 'a') as f:
    f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Loading coordinates file...\n')
    f.flush()

try:
    print("Loading coordinates file...")
    sys.stdout.flush()
    print("Loading coordinates file...", file=sys.stderr)
    sys.stderr.flush()
except:
    pass

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

# Log to file
with open(log_file, 'a') as f:
    f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Number of unique coordinates: {len(df_coordinates_unique)}\n')
    f.flush()

try:
    print(f'Number of unique coordinates: {len(df_coordinates_unique)}')
    sys.stdout.flush()
    print(f'Number of unique coordinates: {len(df_coordinates_unique)}', file=sys.stderr)
    sys.stderr.flush()
except:
    pass

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
geolocator = Nominatim(user_agent="Microbe Atlas metadata project - test - coordinates translation")
geocode_with_rate_limit = RateLimiter(geolocator.reverse, min_delay_seconds=args.min_delay_seconds)

# Check if the results file already exists, load it if it does, else create an empty DataFrame
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
    msg = f'Number of unique coordinates already obtained: {len(results_df)}'
    with open(log_file, 'a') as f:
        f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
        f.flush()
    try:
        print(msg)
        sys.stdout.flush()
        print(msg, file=sys.stderr)
        sys.stderr.flush()
    except:
        pass
else:
    results_df = pd.DataFrame(columns=['latitude', 'longitude', 'place_name'])




# Calculate how many coordinates need to be processed
coords_to_process = []
for index, row in df_coordinates_unique.iterrows():
    if not ((results_df['latitude'] == row['latitude']) & (results_df['longitude'] == row['longitude'])).any():
        coords_to_process.append((index, row))

total_to_process = len(coords_to_process)
msg = f'Number of coordinates to process: {total_to_process}'
with open(log_file, 'a') as f:
    f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
    f.flush()

try:
    print(msg)
    sys.stdout.flush()
    print(msg, file=sys.stderr)
    sys.stderr.flush()
except:
    pass

if total_to_process == 0:
    msg = "All coordinates already processed!"
    with open(log_file, 'a') as f:
        f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
        f.flush()
    try:
        print(msg)
        sys.stdout.flush()
        print(msg, file=sys.stderr)
        sys.stderr.flush()
    except:
        pass
else:
    msg = "Starting reverse geocoding..."
    with open(log_file, 'a') as f:
        f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
        f.flush()
    try:
        print(msg)
        sys.stdout.flush()
        print(msg, file=sys.stderr)
        sys.stderr.flush()
    except:
        pass

    # Initialize progress bar
    log_progress(0, total_to_process, prefix='Geocoding:', suffix='Complete', length=50, log_file_path=log_file)

    batch_size = 100  # Only write every 100 geocodes
    batch_counter = 0 # How many since last write

    avg_time = None   # For ETA

    # Loop to find location name for respective coordinates (saves in batches)
    for n, (index, row) in enumerate(coords_to_process, 1):
        start_time = time.time()

        place_name = reverse_geocode(row['latitude'], row['longitude'])

        new_row = pd.DataFrame([{'latitude': row['latitude'], 'longitude': row['longitude'], 'place_name': place_name}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        batch_counter += 1

        # Time calculations for ETA
        elapsed_time = time.time() - start_time
        if avg_time is None:
            avg_time = elapsed_time
        else:
            avg_time = elapsed_time * 0.3 + avg_time * 0.7

        remaining_items = total_to_process - n
        eta_seconds = remaining_items * avg_time
        eta_minutes = eta_seconds / 60

        # Create suffix with time information
        if eta_minutes > 60:
            eta_str = f"ETA: {eta_minutes/60:.1f}h"
        elif eta_minutes > 1:
            eta_str = f"ETA: {eta_minutes:.1f}m"
        else:
            eta_str = f"ETA: {eta_seconds:.0f}s"
        suffix = f"({n}/{total_to_process}) {eta_str}"

        # Update progress bar
        log_progress(n, total_to_process, prefix='Geocoding:', suffix=suffix, length=50, log_file_path=log_file)

        # Every 10 items, also print a timestamped message
        if n % 10 == 0:
            msg = f"Processed {n}/{total_to_process} coordinates"
            with open(log_file, 'a') as f:
                f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
                f.flush()
            try:
                print(f"\n{msg}")
                sys.stdout.flush()
                print(f"\n{msg}", file=sys.stderr)
                sys.stderr.flush()
            except:
                pass

        # Write batch to file every batch_size items
        if batch_counter >= batch_size:
            results_df.to_csv(output_file, index=False)
            batch_counter = 0

    # Write any remaining results after loop finishes
    if batch_counter > 0:
        results_df.to_csv(output_file, index=False)

    # Final progress bar
    log_progress(total_to_process, total_to_process, prefix='Geocoding:', suffix='Complete!', length=50, log_file_path=log_file)

    msg = "Geocoding completed!"
    with open(log_file, 'a') as f:
        f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
        f.flush()
    try:
        print(f"\n{msg}")
        sys.stdout.flush()
        print(f"\n{msg}", file=sys.stderr)
        sys.stderr.flush()
    except:
        pass

msg = f"Final results: {len(results_df)} coordinates geocoded"
with open(log_file, 'a') as f:
    f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}\n')
    f.flush()

try:
    print(f"\n{msg}")
    sys.stdout.flush()
    print(f"\n{msg}", file=sys.stderr)
    sys.stderr.flush()
    print(results_df[['latitude', 'longitude', 'place_name']])
    sys.stdout.flush()
except:
    pass






# Old script: 

# =============================================================================
# # run as: 
# 
# # python ~/github/metadata_mining/scripts/coord_to_text.py \
# #     --work_dir ~/MicrobeAtlasProject \
# #     --coordinates_file sample.coordinates.reparsed.filtered \
# #     --output_file geocoded_coordinates.csv \
# #     --min_delay_seconds 1.3
# 
# 
# 
# 
# import pandas as pd
# import glob
# import os
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import argparse
# import logging
# from geopy.exc import GeocoderTimedOut
# 
# import sys
# 
# 
# # Set up argument parsing
# parser = argparse.ArgumentParser(description='Read coordinates and translate to location names.')
# parser.add_argument('--work_dir', type=str, required=True, help='work directory')
# parser.add_argument('--coordinates_file', type=str, required=True, help='Christian''s coordinates file')
# parser.add_argument('--output_file', type=str, required=True, help='output filename (to save the translated coordinates)')
# parser.add_argument('--min_delay_seconds', type=float, default=1.5, help='Minimum delay for the geopy rate limiter (seconds)')
# args = parser.parse_args()
# 
# 
# 
# 
# logging.basicConfig(
#     level=logging.ERROR,
#     handlers=[
#         logging.StreamHandler(sys.stderr),
#         logging.FileHandler("/MicrobeAtlasProject/geocoding.log")
#     ]
# )
# 
# 
# 
# 
# # paths 
# work_dir = args.work_dir
# coordinates_file = os.path.join(args.work_dir, args.coordinates_file)
# output_file = os.path.join(args.work_dir, args.output_file)
# 
# 
# # read and filter coordinates file
# df_coordinates_ori = pd.read_csv(coordinates_file, delimiter=' ', header=None, names=['label', 'sample_id', 'latitude', 'longitude'], na_values='None')
# df_coordinates_ori.drop(columns='label', inplace=True)  # Drop the 'label' column if it's not needed
# df_coordinates_ori['latitude'] = pd.to_numeric(df_coordinates_ori['latitude'], errors='coerce')
# df_coordinates_ori['longitude'] = pd.to_numeric(df_coordinates_ori['longitude'], errors='coerce')
# df_coordinates_ori.dropna(subset=['latitude', 'longitude'], inplace=True)
# 
# # filter out invalid latitude or longitude values
# df_coordinates_filtered = df_coordinates_ori[
#     df_coordinates_ori['latitude'].between(-90, 90) & 
#     df_coordinates_ori['longitude'].between(-180, 180)
# ].dropna(subset=['latitude', 'longitude'])
# 
# 
# 
# 
# 
# 
# # drop column
# df_coordinates_filtered = df_coordinates_filtered.drop(columns='sample_id')
# 
# # keep unique coordinates
# df_coordinates_filtered = df_coordinates_filtered.drop_duplicates(subset=['latitude', 'longitude'])
# 
# df_coordinates_unique = df_coordinates_filtered
# 
# print('Number of unique coordinates: ', len(df_coordinates_unique), flush=True)
# #print('Number of unique coordinates: ', len(df_coordinates_unique))
# 
# 
# 
# 
# 
# # Function to perform reverse geocoding
# def reverse_geocode(lat, lon):
#     if not results_df[(results_df['latitude'] == lat) & (results_df['longitude'] == lon)].empty:
#         return results_df[(results_df['latitude'] == lat) & (results_df['longitude'] == lon)]['place_name'].iloc[0]
#     try:
#         location = geocode_with_rate_limit((lat, lon), exactly_one=True, language='en')
#         if location:
#             return location.address
#         else:
#             return None
#     except Exception as e:
#         logging.error(f"Error during geocoding for coordinates ({lat}, {lon}): {e}")
#         return None
#     
#     
#     
# # Initialize Nominatim geocoder with RateLimiter
# geolocator = Nominatim(user_agent="Microbe Atlas metadata project - test - coordinates translation")
# 
# geocode_with_rate_limit = RateLimiter(geolocator.reverse, min_delay_seconds=args.min_delay_seconds)
# 
# 
# # Check if the results file already exists, load it if it does, else create an empty DataFrame
# if os.path.exists(output_file):
#     results_df = pd.read_csv(output_file)
#     print('Number of unique coordinates already obtained: ', len(results_df))
# else:
#     results_df = pd.DataFrame(columns=['latitude', 'longitude', 'place_name'])
# 
# 
# 
# # Loop to find location name for respective coordinates (saves after each hit)
# n = 0
# for index, row in df_coordinates_unique.iterrows():
#     if not ((results_df['latitude'] == row['latitude']) & (results_df['longitude'] == row['longitude'])).any():
#         
#         n += 1
#         #print('Reverse geocoding...', n)
#         
#         
#         from datetime import datetime
#         print(datetime.now(), 'starting geocode', file=sys.stderr, flush=True)
# 
#         print(f'Reverse geocoding... {n}', file=sys.stderr, flush=True)
# 
#         place_name = reverse_geocode(row['latitude'], row['longitude'])
# 
#         print(f"→ Done geocoding {n}", file=sys.stderr, flush=True)
# 
#         
#         new_row = pd.DataFrame([{'latitude': row['latitude'], 'longitude': row['longitude'], 'place_name': place_name}])
#         results_df = pd.concat([results_df, new_row], ignore_index=True)
#         results_df.to_csv(output_file, index=False)
#         
# 
# print(results_df[['latitude', 'longitude', 'place_name']])
# =============================================================================


