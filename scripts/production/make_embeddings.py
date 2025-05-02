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

# Set paths
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")
api_key_path = os.path.join(os.path.expanduser('~'), "Desktop/keys/my_api_key_embeddings")

# Read API key
with open(api_key_path, 'r') as f:
    openai_client = openai.OpenAI(api_key=f.read().strip())

def read_samples(file_path, keywords=False, limit_samples=None):
    samples = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if limit_samples is not None and idx >= limit_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            sample_id, text = parts
            if keywords:
                text = text.strip('{}').replace(',', ' ')
            samples[sample_id] = text.strip()
    return samples

def get_embeddings(samples_dict, batch_size=1000, max_requests_per_round=10, wait_time=30):
    embeddings_dict = {}
    sample_ids = list(samples_dict.keys())
    descriptions = list(samples_dict.values())

    request_count = 0

    for i in range(0, len(descriptions), batch_size):
        chunk = descriptions[i:i + batch_size]
        sample_ids_chunk = sample_ids[i:i + batch_size]
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

            if request_count % max_requests_per_round == 0:
                print(f"Reached {request_count} requests → waiting {wait_time} sec...")
                time.sleep(wait_time)

        except Exception as e:
            print(f"Failed batch {sample_ids_chunk[0]}–{sample_ids_chunk[-1]}: {e}")

    return embeddings_dict

def process_and_save_h5(input_filename, output_filename, keywords=False, limit_samples=None, embedding_dim=1536):
    input_path = os.path.join(work_dir, input_filename)
    output_path = os.path.join(work_dir, output_filename)

    samples = read_samples(input_path, keywords=keywords, limit_samples=limit_samples)
    all_ids = np.array(list(samples.keys()))
    all_texts = np.array(list(samples.values()))

    if os.path.exists(output_path):
        with h5py.File(output_path, 'r+') as h5f:
            existing_ids = h5f['sample_ids'][:]
            existing_ids_set = set(existing_ids)
            new_ids = [sid for sid in all_ids if sid not in existing_ids_set]
            print(f"Found {len(new_ids)} new samples to embed (out of {len(all_ids)} total).")
            if not new_ids:
                print(f"No new samples to embed. File is up-to-date.")
                return
            new_samples = {sid: samples[sid] for sid in new_ids}
            new_embeddings = get_embeddings(new_samples)

            n_new = len(new_embeddings)
            new_ids_array = np.array(list(new_embeddings.keys()))
            new_texts_array = np.array([new_embeddings[sid]['text'] for sid in new_ids_array])
            new_emb_array = np.vstack([new_embeddings[sid]['embedding'] for sid in new_ids_array])

            h5f['sample_ids'].resize(h5f['sample_ids'].shape[0] + n_new, axis=0)
            h5f['texts'].resize(h5f['texts'].shape[0] + n_new, axis=0)
            h5f['embeddings'].resize(h5f['embeddings'].shape[0] + n_new, axis=0)
            h5f['sample_ids'][-n_new:] = new_ids_array
            h5f['texts'][-n_new:] = new_texts_array
            h5f['embeddings'][-n_new:, :] = new_emb_array
            print(f"Updated embeddings saved to: {output_path}")

    else:
        print(f"Creating new HDF5 file: {output_filename}")
        new_embeddings = get_embeddings(samples)
        n_samples = len(new_embeddings)
        ids_array = np.array(list(new_embeddings.keys()))
        texts_array = np.array([new_embeddings[sid]['text'] for sid in ids_array])
        emb_array = np.vstack([new_embeddings[sid]['embedding'] for sid in ids_array])

        with h5py.File(output_path, 'w') as h5f:
            dt = h5py.string_dtype(encoding='utf-8')
            h5f.create_dataset('sample_ids', data=ids_array, maxshape=(None,), dtype=dt)
            h5f.create_dataset('texts', data=texts_array, maxshape=(None,), dtype=dt)
            h5f.create_dataset('embeddings', data=emb_array, maxshape=(None, embedding_dim), dtype='f4')
        print(f"Saved new embeddings to: {output_path}")

def merge_embeddings(subbiomes_h5, keywords_h5, output_h5):
    with h5py.File(subbiomes_h5, 'r') as f_sub, h5py.File(keywords_h5, 'r') as f_key:
        ids_sub = f_sub['sample_ids'][:].astype(str)
        ids_key = f_key['sample_ids'][:].astype(str)

        embeddings_sub = f_sub['embeddings'][:]
        embeddings_key = f_key['embeddings'][:]

        texts_sub = f_sub['texts'][:]
        texts_key = f_key['texts'][:]

        # Find intersection
        common_ids = np.intersect1d(ids_sub, ids_key)
        print(f"Found {len(common_ids)} common samples between subbiomes and keywords.")

        # Build index mappings
        sub_idx_map = {id_: i for i, id_ in enumerate(ids_sub)}
        key_idx_map = {id_: i for i, id_ in enumerate(ids_key)}

        sub_indices = np.array([sub_idx_map[id_] for id_ in common_ids])
        key_indices = np.array([key_idx_map[id_] for id_ in common_ids])

        # Align arrays
        ids_sub_common = ids_sub[sub_indices]
        ids_key_common = ids_key[key_indices]
        emb_sub_common = embeddings_sub[sub_indices]
        emb_key_common = embeddings_key[key_indices]
        texts_sub_common = texts_sub[sub_indices]
        texts_key_common = texts_key[key_indices]

        # Final check
        if not np.array_equal(ids_sub_common, ids_key_common):
            raise ValueError("Aligned sample IDs do not match after indexing!")

        # Average embeddings
        merged_embeddings = (emb_sub_common + emb_key_common) / 2

    with h5py.File(output_h5, 'w') as f_out:
        dt = h5py.string_dtype(encoding='utf-8')
        f_out.create_dataset('sample_ids', data=ids_sub_common.astype('S'), maxshape=(None,), dtype=dt)
        f_out.create_dataset('texts', data=texts_sub_common.astype('S'), maxshape=(None,), dtype=dt)
        f_out.create_dataset('embeddings', data=merged_embeddings, maxshape=(None, emb_sub_common.shape[1]), dtype='f4')

    print(f"Saved merged embeddings to: {output_h5}")



# Run everything
process_and_save_h5('GPT_sub_biomes.txt', 'embeddings/GPT_sub_biomes_embeddings.h5', keywords=False, limit_samples=200000)
process_and_save_h5('GPT_keywords.txt', 'embeddings/GPT_keywords_embeddings.h5', keywords=True, limit_samples=200000)

# Merge embeddings at the end
merge_embeddings(
    os.path.join(work_dir, 'embeddings/GPT_sub_biomes_embeddings.h5'),
    os.path.join(work_dir, 'embeddings/GPT_keywords_embeddings.h5'),
    os.path.join(work_dir, 'embeddings/GPT_sub_biomes_keywords_embeddings.h5')
)







def count_samples_in_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        n_samples = f['sample_ids'].shape[0]
        print(f"✅ {n_samples} samples in {h5_path}")

# Example usage:
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")

count_samples_in_h5(os.path.join(work_dir, 'embeddings/GPT_sub_biomes_embeddings.h5'))
count_samples_in_h5(os.path.join(work_dir, 'embeddings/GPT_keywords_embeddings.h5'))
count_samples_in_h5(os.path.join(work_dir, 'embeddings/GPT_sub_biomes_keywords_embeddings.h5'))








