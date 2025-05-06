#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:04:46 2025

@author: danielagaio
"""


import pandas as pd
from pathlib import Path

# Set directories
work_dir = Path.home() / "cloudstor" / "Gaio" / "MicrobeAtlasProject"
production_dir = work_dir / "production"
hackathon_dir = work_dir / "Hackathon"

# Make sure Hackathon directory exists
hackathon_dir.mkdir(exist_ok=True)

# Output files
output_files = {
    "biomes": hackathon_dir / "GPT_biomes.txt",
    "geo_texts": hackathon_dir / "GPT_geo_texts.txt",
    "keywords": hackathon_dir / "GPT_keywords.txt",
    "sub_biomes": hackathon_dir / "GPT_sub_biomes.txt",
}

# Load existing data if available
existing_data = {}
for key, filepath in output_files.items():
    if filepath.exists():
        existing_data[key] = pd.read_csv(filepath, sep="\t", header=None, names=["sample_id", key])
    else:
        existing_data[key] = pd.DataFrame(columns=["sample_id", key])

# Collect all gpt_clean_output*.csv files
csv_files = list(production_dir.glob("gpt_clean_output*.csv"))

# Read and combine all CSVs
dfs = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Prepare new dataframes for each output
new_data = {
    "biomes": combined_df[["sample_id", "biome_label"]].rename(columns={"biome_label": "biomes"}),
    "geo_texts": combined_df[["sample_id", "geo_location"]].rename(columns={"geo_location": "geo_texts"}),
    "keywords": combined_df[["sample_id", "keywords"]].rename(columns={"keywords": "keywords"}),
    "sub_biomes": combined_df[["sample_id", "sub_biome"]].rename(columns={"sub_biome": "sub_biomes"}),
}

# Update existing data and save
for key, new_df in new_data.items():
    combined = pd.concat([existing_data[key], new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_id"])  # Keep first occurrence
    combined.to_csv(output_files[key], sep="\t", header=False, index=False)

print("Hackathon files updated successfully!")





# #####
# # Extra to know how many NAs in origianl files per field: 
# 
# import pandas as pd
# from pathlib import Path
# 
# # Set directory
# work_dir = Path.home() / "cloudstor" / "Gaio" / "MicrobeAtlasProject"
# production_dir = work_dir / "production"
# 
# # Collect all gpt_clean_output*.csv files
# csv_files = list(production_dir.glob("gpt_clean_output*.csv"))
# 
# # Columns to check
# columns_to_check = ['biome_label', 'geo_location', 'keywords', 'sub_biome']
# 
# # Initialize total counters
# total_counts = {col: 0 for col in columns_to_check}
# total_rows = 0
# 
# # Process each file
# print("\nPer-file NA summary:")
# for f in csv_files:
#     df = pd.read_csv(f)
#     file_total = len(df)
#     total_rows += file_total
#     file_counts = {}
#     for col in columns_to_check:
#         if col in df.columns:
#             na_count = df[col].isna().sum()
#             percent = (na_count / file_total * 100) if file_total > 0 else 0
#             file_counts[col] = f"{na_count} / {file_total} ({percent:.2f}%)"
#             total_counts[col] += na_count
#         else:
#             file_counts[col] = 'MISSING'
#     # Print per-file summary
#     print(f"{f.name}: {file_counts}")
# 
# # Print total summary
# print("\nTotal NA counts across all files:")
# for col in columns_to_check:
#     na_count = total_counts[col]
#     percent = (na_count / total_rows * 100) if total_rows > 0 else 0
#     print(f"{col}: {na_count} / {total_rows} ({percent:.2f}%)")
# #####
# =============================================================================