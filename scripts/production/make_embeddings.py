#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:35:04 2025

@author: danielagaio
"""



# =============================================================================
# import os
# import openai
# import json
# 
# # Set paths
# work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")
# api_key_path = os.path.join(os.path.expanduser('~'), "Desktop/keys/my_api_key_embeddings")
# 
# # Read API key
# with open(api_key_path, 'r') as f:
#     openai_client = openai.OpenAI(api_key=f.read().strip())
# 
# # Function to read samples from a TXT file
# def read_samples(file_path, keywords=False, limit_samples=None):
#     samples = {}
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for idx, line in enumerate(f):
#             if limit_samples is not None and idx >= limit_samples:
#                 break
#             parts = line.strip().split('\t')
#             if len(parts) != 2:
#                 continue
#             sample_id, text = parts
#             if keywords:
#                 text = text.strip('{}').replace(',', ' ')  # Remove {} and commas for keyword lists
#             samples[sample_id] = text.strip()
#     return samples
# 
# # Function to get embeddings
# def get_embeddings(samples_dict):
#     embeddings_dict = {}
#     batch_size = 200
#     sample_ids = list(samples_dict.keys())
#     descriptions = list(samples_dict.values())
# 
#     for i in range(0, len(descriptions), batch_size):
#         chunk = descriptions[i:i+batch_size]
#         sample_ids_chunk = sample_ids[i:i+batch_size]
#         try:
#             response = openai_client.embeddings.create(
#                 input=chunk,
#                 model="text-embedding-3-small"
#             )
#             embeddings = [item.embedding for item in response.data]
#             for j, sample_id in enumerate(sample_ids_chunk):
#                 embeddings_dict[sample_id] = {
#                     'embedding': embeddings[j],
#                     'text': samples_dict[sample_id]
#                 }
#         except Exception as e:
#             print(f"Failed batch {sample_ids_chunk[0]} - {sample_ids_chunk[-1]}: {e}")
# 
#     return embeddings_dict
# 
# # Process a file and save embeddings
# def process_and_save(input_filename, output_filename, keywords=False, limit_samples=None):
#     input_path = os.path.join(work_dir, input_filename)
#     output_path = os.path.join(work_dir, output_filename)
# 
#     # Load existing embeddings if file exists
#     if os.path.exists(output_path):
#         print(f"Loading existing embeddings: {output_filename}")
#         with open(output_path, 'r', encoding='utf-8') as f:
#             existing_embeddings = json.load(f)
#     else:
#         existing_embeddings = {}
# 
#     print(f"Processing: {input_filename}")
#     samples = read_samples(input_path, keywords=keywords, limit_samples=limit_samples)
# 
#     # Identify samples that still need embeddings
#     samples_to_embed = {sid: desc for sid, desc in samples.items() if sid not in existing_embeddings}
#     print(f"Found {len(samples_to_embed)} new samples to embed (out of {len(samples)} total samples).")
# 
#     # If there are new samples, get their embeddings
#     if samples_to_embed:
#         new_embeddings = get_embeddings(samples_to_embed)
#         existing_embeddings.update(new_embeddings)
# 
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(existing_embeddings, f, ensure_ascii=False, indent=4)
# 
#         print(f"Updated embeddings saved to: {output_path}")
#     else:
#         print(f"No new samples to embed. Existing file is up-to-date.")
# 
# # Run everything
# process_and_save('GPT_sub_biomes.txt', 'embeddings/GPT_sub_biomes_embeddings.json', keywords=False, limit_samples=200000)
# process_and_save('GPT_keywords.txt', 'embeddings/GPT_keywords_embeddings.json', keywords=True, limit_samples=200000)
# 
# =============================================================================





import os
import ijson
import h5py
import numpy as np

def convert_json_to_h5_single_pass(json_path, h5_path, embedding_dim=1536, chunk_size=10000):
    with h5py.File(h5_path, 'w') as h5f:
        dt = h5py.string_dtype(encoding='utf-8')
        dset_ids = h5f.create_dataset('sample_ids', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)
        dset_texts = h5f.create_dataset('texts', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)
        dset_embeddings = h5f.create_dataset('embeddings', shape=(0, embedding_dim), maxshape=(None, embedding_dim), dtype='f', chunks=True)

        ids_chunk, texts_chunk, embeddings_chunk = [], [], []
        idx = 0

        with open(json_path, 'rb') as f:
            parser = ijson.kvitems(f, '')
            for sample_id, record in parser:
                embedding = np.array(record['embedding'], dtype='f')
                text = record['text']
                ids_chunk.append(sample_id)
                texts_chunk.append(text)
                embeddings_chunk.append(embedding)
                idx += 1

                if idx % chunk_size == 0:
                    # Extend datasets
                    dset_ids.resize(dset_ids.shape[0] + chunk_size, axis=0)
                    dset_texts.resize(dset_texts.shape[0] + chunk_size, axis=0)
                    dset_embeddings.resize(dset_embeddings.shape[0] + chunk_size, axis=0)
                    # Write chunk
                    dset_ids[-chunk_size:] = ids_chunk
                    dset_texts[-chunk_size:] = texts_chunk
                    dset_embeddings[-chunk_size:, :] = np.vstack(embeddings_chunk)
                    print(f"Processed {idx} samples...")
                    # Reset chunk lists
                    ids_chunk, texts_chunk, embeddings_chunk = [], [], []

        # Write any remaining data
        if ids_chunk:
            final_chunk_size = len(ids_chunk)
            dset_ids.resize(dset_ids.shape[0] + final_chunk_size, axis=0)
            dset_texts.resize(dset_texts.shape[0] + final_chunk_size, axis=0)
            dset_embeddings.resize(dset_embeddings.shape[0] + final_chunk_size, axis=0)
            dset_ids[-final_chunk_size:] = ids_chunk
            dset_texts[-final_chunk_size:] = texts_chunk
            dset_embeddings[-final_chunk_size:, :] = np.vstack(embeddings_chunk)
            print(f"Processed {idx} samples...")

    print(f"Conversion complete → {h5_path}")

# Example usage:
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon")

convert_json_to_h5_single_pass(
    os.path.join(work_dir, 'embeddings/GPT_keywords_embeddings.json'),
    os.path.join(work_dir, 'embeddings/GPT_keywords_embeddings.h5')
)

convert_json_to_h5_single_pass(
    os.path.join(work_dir, 'embeddings/GPT_sub_biomes_embeddings.json'),
    os.path.join(work_dir, 'embeddings/GPT_sub_biomes_embeddings.h5')
)


