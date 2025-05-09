#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:11:20 2025

@author: danielagaio
"""


import os
import random
import argparse

# --------- Argument parser ---------
parser = argparse.ArgumentParser(description="Generate sampling list per biome")
parser.add_argument("--biome_file", type=str, default="/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt",
                    help="Path to GPT_biomes.txt")
parser.add_argument("--nspb", type=int, required=True,
                    help="Number of samples per biome")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--output_dir", type=str, default="/Users/danielagaio/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings/sampling",
                    help="Directory to save the sampling list")

args = parser.parse_args()

# --------- Biomes of interest ---------
biomes_of_interest = {'animal', 'plant', 'water', 'soil', 'other'}

# --------- Load biome labels ---------
print(f"Loading biome labels from {args.biome_file}...")
biome_to_ids = {}
with open(args.biome_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            sid, biome = parts
            if biome in biomes_of_interest:
                if biome not in biome_to_ids:
                    biome_to_ids[biome] = []
                biome_to_ids[biome].append(sid)

print(f"Found {len(biome_to_ids)} biomes to include: {list(biome_to_ids.keys())}")

# --------- Check available counts ---------
warnings = []
for biome, ids in biome_to_ids.items():
    if len(ids) < args.nspb:
        warnings.append(f"⚠️ Biome '{biome}' has only {len(ids)} samples but you requested {args.nspb}.")

if warnings:
    print("\n".join(warnings))
    response = input("⚠️ Are you sure you want to continue? (y/n): ")
    if response.strip().lower() != 'y':
        print("❌ Aborted by user due to insufficient samples.")
        exit(1)

# --------- Apply sampling ---------
random.seed(args.seed)
final_selected_ids = []

for biome, ids in biome_to_ids.items():
    if len(ids) <= args.nspb:
        chosen = ids
    else:
        chosen = random.sample(ids, args.nspb)
    final_selected_ids.extend(chosen)
    print(f"{biome}: selected {len(chosen)} samples (from {len(ids)} available)")

print(f"Total selected samples: {len(final_selected_ids)}")

# --------- Prepare output ---------
os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(
    args.output_dir,
    f"sampling_nspb{args.nspb}_seed{args.seed}.txt"
)

with open(output_file, 'w') as f:
    for sid in final_selected_ids:
        f.write(f"{sid}\n")

print(f"✅ Sampling list saved to {output_file}")

