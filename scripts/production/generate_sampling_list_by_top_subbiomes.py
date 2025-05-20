#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:55:39 2025

@author: danielagaio
"""

import os
import random
import argparse
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# --------- Argument parser ---------
parser = argparse.ArgumentParser(description="Generate sampling list using top-N sub-biomes per biome")
parser.add_argument("--biome_file", type=str, required=True,
                    help="Path to GPT_biomes.txt (tab-separated: sample_id<TAB>biome)")
parser.add_argument("--subbiome_file", type=str, required=True,
                    help="Path to GPT_sub_biomes.txt (tab-separated: sample_id<TAB>subbiome)")
parser.add_argument("--top_n", type=int, required=True,
                    help="Top N most common sub-biomes per biome to include")
parser.add_argument("--k", type=int, default=100,
                    help="Number of samples per sub-biome")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--output_dir", type=str, default="./sampling",
                    help="Directory to save the sampling list")
args = parser.parse_args()

# --------- Load biome labels ---------
print(f"📂 Loading biome labels from {args.biome_file}...")
sid_to_biome = {}
with open(args.biome_file, 'r') as f:
    for line in f:
        sid, biome = line.strip().split('\t')
        sid_to_biome[sid] = biome

# --------- Load sub-biome labels ---------
print(f"📂 Loading sub-biome labels from {args.subbiome_file}...")
sid_to_subbiome = {}
with open(args.subbiome_file, 'r') as f:
    for line in f:
        
        
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue  # skip bad lines
        sid, sub = parts

        
        sid_to_subbiome[sid] = sub

# --------- Group samples per (biome → sub-biome → [sample_ids]) ---------
biome_to_subbiome_to_ids = defaultdict(lambda: defaultdict(list))

# --------- Define biomes of interest ---------
target_biomes = {"animal", "plant", "soil", "water", "other"}

# --------- Group by biome → subbiome ---------
for sid, biome in sid_to_biome.items():
    if biome not in target_biomes:
        continue
    if sid not in sid_to_subbiome:
        continue
    subbiome = sid_to_subbiome[sid]
    biome_to_subbiome_to_ids[biome][subbiome].append(sid)


# --------- Sampling ---------
random.seed(args.seed)
final_selected_ids = []

print(f"\n📊 Sampling top {args.top_n} sub-biomes per biome (with {args.k} samples each):\n")



def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# --------- Sampling with label de-duplication ---------
random.seed(args.seed)
final_selected_ids = []

print(f"\n📊 Sampling top {args.top_n} sub-biomes per biome (with {args.k} samples each, avoiding duplicates):\n")

# --------- Sampling with word overlap exclusion ---------
random.seed(args.seed)
final_selected_ids = []

print(f"\n📊 Sampling top {args.top_n} sub-biomes per biome (with {args.k} samples each, no word repetition):\n")

for biome, sub_dict in biome_to_subbiome_to_ids.items():
    sub_counts = {sub: len(ids) for sub, ids in sub_dict.items()}
    sorted_subs = sorted(sub_counts.items(), key=lambda x: -x[1])

    used_words = set()
    selected_subs = []

    print(f"Biome: {biome}")
    for sub, count in sorted_subs:
        words = set(sub.lower().split())
        if words & used_words:
            continue  # skip if any word overlaps
        used_words.update(words)
        selected_subs.append(sub)

        sids = sub_dict[sub]
        chosen = sids if len(sids) <= args.k else random.sample(sids, args.k)
        final_selected_ids.extend(chosen)
        print(f"  - {sub}: selected {len(chosen)} / available {len(sids)}")

        if len(selected_subs) >= args.top_n:
            break

print(f"\n✅ Total selected samples: {len(final_selected_ids)}")


# --------- Save to file ---------
os.makedirs(args.output_dir, exist_ok=True)
out_file = os.path.join(
    args.output_dir,
    f"sampling_topnspb{args.top_n}_k{args.k}_seed{args.seed}.txt"
)

with open(out_file, 'w') as f:
    for sid in final_selected_ids:
        f.write(f"{sid}\n")

print(f"\n📄 Sampling list saved to: {out_file}")




  
# python generate_sampling_list_by_top_subbiomes.py \
#   --biome_file ~/Desktop/MicrobeAtlasProject/Hackathon/GPT_biomes.txt \
#   --subbiome_file ~/Desktop/MicrobeAtlasProject/Hackathon/GPT_sub_biomes.txt \
#   --top_n 50 \
#   --k 100 \
#   --seed 42 \
#   --output_dir ~/Desktop/MicrobeAtlasProject/Hackathon/embeddings/sampling/


