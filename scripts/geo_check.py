#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:17:28 2024

@author: dgaio
"""




import pandas as pd
import glob
import os


# paths
work_dir = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject'

# to get gpt files:
file_pattern = os.path.join(work_dir, '/production/gpt_clean_output*.csv')
files = glob.glob(file_pattern)

# to get sample ids - coordinates 
coordinates_file = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.coordinates.reparsed.filtered'

# to get coordinates to text
geocoded_coordinates.csv


# 1. Concatenate all gpt files
gpt_geo_text = pd.concat(
    (pd.read_csv(f, usecols=['sample_id', 'geo_location']) for f in files),
    ignore_index=True
).dropna(subset=['geo_location'])














# # for testing:
##geolocator.reverse(('31.110000', '121.380000'), exactly_one=True)






# =============================================================================
# 
# # APPEND FICTIVE INFO (MISMATCHES)
# 
# import pandas as pd
# 
# # Assuming df_coordinates_unique is already defined
# # Create two fictive rows with mismatching geo_location and place_name
# fictive_data = [
#     {
#         'sample_id': 'SRS9999999',
#         'geo_location': 'Lima, Peru',
#         'latitude': -12.0464,
#         'longitude': -77.0428,
#         'place_name': 'Moscow, Russia'  # Completely different location
#     },
#     {
#         'sample_id': 'SRS8888888',
#         'geo_location': 'Sydney, Australia',
#         'latitude': -33.8688,
#         'longitude': 151.2093,
#         'place_name': 'New York, USA'  # Completely different location
#     }
# ]
# 
# # Convert the fictive data into a DataFrame
# fictive_df = pd.DataFrame(fictive_data)
# 
# # Append the fictive rows to the existing DataFrame
# df_coordinates_unique = df_coordinates_unique.append(fictive_df, ignore_index=True)
# 
# 
# 
# 
# import os
# import openai
# import json
# import openai
# import numpy as np
# import pandas as pd
# 
# # Set API key path and initialize OpenAI API
# api_key_path = '/Users/dgaio/my_api_key_embeddings'
# with open(api_key_path, "r") as file:
#     openai.api_key = file.read().strip()
# 
# # Function to get embeddings from OpenAI API
# def get_embedding(text):
#     try:
#         response = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")
#         return response['data'][0]['embedding']
#     except Exception as e:
#         print(f"Error retrieving embedding for text '{text}': {e}")
#         return None
# 
# # Function to calculate cosine similarity between two embeddings
# def cosine_similarity(embedding1, embedding2):
#     return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
# 
# # Function to compare geo_location and place_name using embeddings
# def compare_geo_location_with_place_name(row):
#     geo_loc_embedding = get_embedding(row['geo_location'])
#     place_name_embedding = get_embedding(row['place_name'])
#     
#     if geo_loc_embedding is not None and place_name_embedding is not None:
#         similarity_score = cosine_similarity(geo_loc_embedding, place_name_embedding)
#         return similarity_score >= 0.5, similarity_score
#     else:
#         return False, None
# 
# 
# df = df_coordinates_unique
# 
# # Apply the comparison function to each row
# df[['is_similar', 'similarity_score']] = df.apply(compare_geo_location_with_place_name, axis=1, result_type='expand')
# 
# # Print the results
# print(df[['sample_id', 'geo_location', 'place_name', 'is_similar', 'similarity_score']])
# 
# 
# 
# import os
# import openai
# import numpy as np
# import pandas as pd
# 
# # Set API key path and initialize OpenAI API
# api_key_path = '/Users/dgaio/my_api_key_embeddings'
# with open(api_key_path, "r") as file:
#     openai.api_key = file.read().strip()
# 
# # Function to get embeddings from OpenAI API with juxtaposition
# def get_embedding(text):
#     try:
#         # Adding a juxtaposition context to the input text
#         response = openai.Embedding.create(input=[f"A dog in {text}"], engine="text-embedding-ada-002")
#         return response['data'][0]['embedding']
#     except Exception as e:
#         print(f"Error retrieving embedding for text '{text}': {e}")
#         return None
# 
# # Function to calculate cosine similarity between two embeddings
# def cosine_similarity(embedding1, embedding2):
#     return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
# 
# # Function to compare geo_location and place_name using embeddings with juxtaposition
# def compare_geo_location_with_place_name(row):
#     # Adding the context "A dog in" to both geo_location and place_name
#     geo_loc_embedding = get_embedding(f"A dog in {row['geo_location']}")
#     place_name_embedding = get_embedding(f"A dog in {row['place_name']}")
#     
#     if geo_loc_embedding is not None and place_name_embedding is not None:
#         similarity_score = cosine_similarity(geo_loc_embedding, place_name_embedding)
#         return similarity_score >= 0.5, similarity_score
#     else:
#         return False, None
# 
# # Assuming df_coordinates_unique is your DataFrame
# df = df_coordinates_unique
# 
# # Apply the comparison function to each row
# df[['is_similar', 'similarity_score']] = df.apply(compare_geo_location_with_place_name, axis=1, result_type='expand')
# 
# # Print the results
# print(df[['sample_id', 'geo_location', 'place_name', 'is_similar', 'similarity_score']])
# 
# 
# =============================================================================


