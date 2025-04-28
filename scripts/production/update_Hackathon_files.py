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
    "biomes": hackathon_dir / "biomes.txt",
    "geo_texts": hackathon_dir / "geo_texts.txt",
    "keywords": hackathon_dir / "keywords.txt",
    "sub_biomes": hackathon_dir / "sub_biomes.txt",
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
