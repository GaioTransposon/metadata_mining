#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:35:04 2025

@author: danielagaio
"""



import os
import openai
import h5py
import numpy as np
import time
import json
import itertools




# ===== CONFIGURATION =====
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")
api_key_path = os.path.join(os.path.expanduser('~'), "Desktop/keys/my_api_key_embeddings")
input_files = [
    ('GPT_sub_biomes.txt', 'embeddings/GPT_sub_biomes_embeddings.h5', 'state_file_sub_biomes.txt', False),
    ('GPT_keywords.txt', 'embeddings/GPT_keywords_embeddings.h5', 'state_file_keywords.txt', True)
]

batch_size = 1000  # API batch size
file_slice_size = 10000  # how many samples to load per slice
max_requests_per_round = 100  # requests before waiting
wait_time = 60  # seconds to wait
embedding_dim = 1536

# ===== SETUP API =====
with open(api_key_path, 'r') as f:
    openai_client = openai.OpenAI(api_key=f.read().strip())

# ===== HELPER FUNCTIONS =====
def read_samples_slice(file_path, start, end, keywords=False):
    samples = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in itertools.islice(f, start, end):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            sample_id, text = parts
            if keywords:
                text = text.strip('{}').replace(',', ' ')
            samples[sample_id] = text.strip()
    return samples

def get_embeddings(samples_dict):
    embeddings_dict = {}
    sample_ids = list(samples_dict.keys())
    descriptions = list(samples_dict.values())

    request_count = 0
    for i in range(0, len(descriptions), batch_size):
        chunk = descriptions[i:i + batch_size]
        sample_ids_chunk = sample_ids[i:i + batch_size]
        start_time = time.time()
        try:
            response = openai_client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            )
            embeddings = [item.embedding for item in response.data]
            for j, sample_id in enumerate(sample_ids_chunk):
                embeddings_dict[sample_id] = {
                    'embedding': embeddings[j],
                    'text': samples_dict[sample_id]
                }
            request_count += 1
            elapsed = time.time() - start_time
            print(f" → Batch {request_count} ({len(sample_ids_chunk)} samples) took {elapsed:.2f}s")
            if request_count % max_requests_per_round == 0:
                print(f" → Reached {request_count} requests, waiting {wait_time}s...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Failed batch {sample_ids_chunk[0]}–{sample_ids_chunk[-1]}: {e}")
    return embeddings_dict

def update_state_file(state_file, index):
    with open(state_file, 'w') as f:
        json.dump({'last_sample': index}, f)

def get_current_index(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            return state.get('last_sample', 0)
    return 0

def process_file(input_file, output_file, state_file, keywords):
    input_path = os.path.join(work_dir, input_file)
    output_path = os.path.join(work_dir, output_file)
    state_path = os.path.join(work_dir, 'embeddings', state_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)

    start_idx = get_current_index(state_path)
    total_processed = 0
    total_start_time = time.time()

    while True:
        slice_start = start_idx
        slice_end = slice_start + file_slice_size
        samples = read_samples_slice(input_path, slice_start, slice_end, keywords)
        if not samples:
            print(f"✅ All samples processed for {input_file}")
            break
        print(f"Processing samples {slice_start}–{slice_end -1} ({len(samples)})")

        slice_start_time = time.time()
        embeddings = get_embeddings(samples)
        if not embeddings:
            print("⚠️ No embeddings generated, skipping slice")
            break

        ids = list(embeddings.keys())
        texts = [embeddings[s]['text'] for s in ids]
        emb_array = np.vstack([embeddings[s]['embedding'] for s in ids])
        dt = h5py.string_dtype(encoding='utf-8')

        if os.path.exists(output_path):
            with h5py.File(output_path, 'r+') as h5f:
                for name, data, shape in [('sample_ids', ids, (None,)), ('texts', texts, (None,)), ('embeddings', emb_array, (None, embedding_dim))]:
                    if name not in h5f:
                        maxshape = (None,) if name != 'embeddings' else (None, embedding_dim)
                        dtype = dt if name != 'embeddings' else 'f4'
                        h5f.create_dataset(name, data=data, maxshape=maxshape, dtype=dtype)
                    else:
                        h5f[name].resize(h5f[name].shape[0] + len(ids), axis=0)
                        h5f[name][-len(ids):] = data
        else:
            with h5py.File(output_path, 'w') as h5f:
                h5f.create_dataset('sample_ids', data=ids, maxshape=(None,), dtype=dt)
                h5f.create_dataset('texts', data=texts, maxshape=(None,), dtype=dt)
                h5f.create_dataset('embeddings', data=emb_array, maxshape=(None, embedding_dim), dtype='f4')
        
        slice_elapsed = time.time() - slice_start_time
        print(f" → Slice done in {slice_elapsed/60:.2f} min")

        start_idx = slice_end
        total_processed += len(ids)
        update_state_file(state_path, start_idx)
    
    total_elapsed = time.time() - total_start_time
    print(f"✅ Finished {input_file}: {total_processed} samples in {total_elapsed/60:.2f} min")

# ===== RUN PIPELINE =====
overall_start_time = time.time()

for infile, outfile, statefile, keywords in input_files:
    process_file(infile, outfile, statefile, keywords)
    print(f"✅ Completed {infile}\n")

overall_elapsed = time.time() - overall_start_time
print(f"🏁 All embedding runs completed in {overall_elapsed/60:.2f} minutes")

# 🏁 speed for keywords: : 1076410 samples in 60.78 min



####


# averaged embeddings of sub-biomes + keywords 

import os
import numpy as np
import h5py

# Paths
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")

subbiomes_path = os.path.join(work_dir, 'embeddings/GPT_sub_biomes_embeddings.h5')
keywords_path = os.path.join(work_dir, 'embeddings/GPT_keywords_embeddings.h5')
output_path = os.path.join(work_dir, 'embeddings/GPT_sub_biomes_keywords_embeddings.h5')

batch_size = 10000

# ===== Safeguard: avoid overwriting output =====
if os.path.exists(output_path):
    print(f"⚠️ Output file {output_path} already exists. Exiting to avoid overwrite.")
    exit(1)

# ===== Load input file metadata =====
with h5py.File(subbiomes_path, 'r') as subf, h5py.File(keywords_path, 'r') as keyf:
    sub_ids = subf['sample_ids'][:]
    key_ids = keyf['sample_ids'][:]

    # Decode bytes to strings
    sub_ids = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sub_ids])
    key_ids = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in key_ids])

    # Find intersection and get index positions
    common_ids, sub_idx_pos, key_idx_pos = np.intersect1d(sub_ids, key_ids, return_indices=True)
    print(f"✅ {len(common_ids)} common samples found")

    # Sort indices ONCE (use subbiome order)
    sorted_order = np.argsort(sub_idx_pos)
    common_ids = common_ids[sorted_order]
    sub_idx_pos = sub_idx_pos[sorted_order]
    key_idx_pos = key_idx_pos[np.argsort(key_idx_pos)]  # align keywords

    # ===== Prepare output HDF5 file =====
    dt = h5py.string_dtype(encoding='utf-8')
    with h5py.File(output_path, 'w') as outf:
        outf.create_dataset('sample_ids', shape=(0,), maxshape=(None,), dtype=dt)
        outf.create_dataset('sub_texts', shape=(0,), maxshape=(None,), dtype=dt)
        outf.create_dataset('key_texts', shape=(0,), maxshape=(None,), dtype=dt)
        outf.create_dataset('embeddings', shape=(0, 1536), maxshape=(None, 1536), dtype='f4')

        for i in range(0, len(common_ids), batch_size):
            batch_ids = common_ids[i:i + batch_size]
            sub_batch_idx = sub_idx_pos[i:i + batch_size]
            key_batch_idx = key_idx_pos[i:i + batch_size]

            sub_texts = subf['texts'][sub_batch_idx]
            key_texts = keyf['texts'][key_batch_idx]
            sub_embeds = subf['embeddings'][sub_batch_idx]
            key_embeds = keyf['embeddings'][key_batch_idx]

            # Decode texts if needed
            sub_texts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sub_texts]
            key_texts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in key_texts]

            avg_embeds = (sub_embeds + key_embeds) / 2

            # Resize and write
            n = outf['sample_ids'].shape[0]
            outf['sample_ids'].resize(n + len(batch_ids), axis=0)
            outf['sub_texts'].resize(n + len(batch_ids), axis=0)
            outf['key_texts'].resize(n + len(batch_ids), axis=0)
            outf['embeddings'].resize(n + len(batch_ids), axis=0)

            outf['sample_ids'][n:] = np.array(batch_ids, dtype=object)
            outf['sub_texts'][n:] = np.array(sub_texts, dtype=object)
            outf['key_texts'][n:] = np.array(key_texts, dtype=object)
            outf['embeddings'][n:] = avg_embeds

            print(f" → Processed batch {i // batch_size +1} ({len(batch_ids)} samples)")

print(f"✅ Saved averaged embeddings to {output_path}")


####


# create file of sample ids intersection across 3 h5 files: 
    

# --------- Helper: Load sample IDs only ---------
def get_sample_ids(filepath):
    with h5py.File(filepath, 'r') as f:
        sample_ids_raw = f['sample_ids'][:]
        sample_ids = sample_ids_raw.astype(str)
    return set(sample_ids)

# --------- Paths ---------
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings")

subbiomes_path = os.path.join(work_dir, 'GPT_sub_biomes_embeddings.h5')
keywords_path = os.path.join(work_dir, 'GPT_keywords_embeddings.h5')
sb_keywords_path = os.path.join(work_dir, 'GPT_sub_biomes_keywords_embeddings.h5')

output_path = os.path.join(work_dir, 'common_sample_ids_of_embeddings')

# --------- Compute intersection ---------
print("Loading sample IDs...")
ids_subbiomes = get_sample_ids(subbiomes_path)
ids_sb_keywords = get_sample_ids(sb_keywords_path)
ids_keywords = get_sample_ids(keywords_path)

common_sample_ids = ids_subbiomes & ids_sb_keywords & ids_keywords
print(f"Common sample IDs across all files: {len(common_sample_ids)}")

# --------- Save to disk ---------
common_sample_ids_array = np.array(list(common_sample_ids))
np.save(output_path, common_sample_ids_array)
print(f"Saved intersection list to {output_path}")



####



def count_samples_in_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        sample_ids = f['sample_ids'][:]
        n_samples = sample_ids.shape[0]
        unique_sample_ids = np.unique(sample_ids)
        n_unique_samples = unique_sample_ids.shape[0]
        print(f"✅ {n_samples} samples in {h5_path}")
        print(f"🔹 {n_unique_samples} unique samples in {h5_path}")

# Example usage:
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")

count_samples_in_h5(os.path.join(work_dir, 'embeddings/GPT_sub_biomes_embeddings.h5')) # 2036583
count_samples_in_h5(os.path.join(work_dir, 'embeddings/GPT_keywords_embeddings.h5')) # 2056410
count_samples_in_h5(os.path.join(work_dir, 'embeddings/GPT_sub_biomes_keywords_embeddings.h5')) # 2036583




####




