#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:36:38 2026

@author: dgaio
"""



# run as: 
# python  ~/github/metadata_mining/scripts/production/align_and_average_embeddings.py \
#   --work_dir ~/MicrobeAtlasProject2024/production \
#   --embedding_dim 1536

# python  ~/github/metadata_mining/scripts/production/align_and_average_embeddings.py \
#   --work_dir ~/MicrobeAtlasProject2024/production \
#   --embedding_dim 3072





import os
import time
import argparse
import h5py
import numpy as np


# ===== ARGUMENTS =====
parser = argparse.ArgumentParser()

parser.add_argument("--work_dir", default=os.path.join(os.path.expanduser("~"), "MicrobeAtlasProject2024/production"))
parser.add_argument("--embedding_dim", type=int, default=3072)
parser.add_argument("--batch_size", type=int, default=10000)

parser.add_argument("--subbiomes_input", default=None)
parser.add_argument("--keywords_input", default=None)

parser.add_argument("--aligned_output", default=None)
parser.add_argument("--combined_output", default=None)

args = parser.parse_args()


# ===== CONFIGURATION =====
work_dir = args.work_dir
embedding_dim = args.embedding_dim
batch_size = args.batch_size

subbiomes_path = (
    os.path.join(work_dir, args.subbiomes_input)
    if args.subbiomes_input
    else os.path.join(work_dir, f"embeddings/GPT_sub_biomes_embeddings_{embedding_dim}.h5")
)

keywords_path = (
    os.path.join(work_dir, args.keywords_input)
    if args.keywords_input
    else os.path.join(work_dir, f"embeddings/GPT_keywords_embeddings_{embedding_dim}.h5")
)

aligned_output = (
    os.path.join(work_dir, args.aligned_output)
    if args.aligned_output
    else os.path.join(work_dir, f"embeddings/GPT_sub_biomes_embeddings_aligned_{embedding_dim}.h5")
)

combined_output = (
    os.path.join(work_dir, args.combined_output)
    if args.combined_output
    else os.path.join(work_dir, f"embeddings/GPT_sub_biomes_keywords_embeddings_{embedding_dim}.h5")
)


# ===== HELPER FUNCTIONS =====
def decode_array(arr):
    return np.array([
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in arr
    ])


def count_samples_in_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        sample_ids = f["sample_ids"][:]
        n_samples = sample_ids.shape[0]
        n_unique_samples = np.unique(sample_ids).shape[0]

        print(f"✅ {n_samples} samples in {h5_path}")
        print(f"🔹 {n_unique_samples} unique samples in {h5_path}")


def check_input_file(h5_path, label):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{label} file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        for dataset in ["sample_ids", "texts", "embeddings"]:
            if dataset not in f:
                raise KeyError(f"{label} file is missing dataset: {dataset}")

        actual_dim = f["embeddings"].shape[1]
        if actual_dim != embedding_dim:
            raise ValueError(
                f"{label} embedding dimension is {actual_dim}, "
                f"but --embedding_dim is {embedding_dim}"
            )


# ===== STEP 1: ALIGN SUB-BIOME EMBEDDINGS TO KEYWORD ORDER =====
def align_subbiomes_to_keywords():
    if os.path.exists(aligned_output):
        print(f"⚠️ Aligned file already exists, skipping: {aligned_output}")
        return

    print("🔁 Aligning sub-biome embeddings to keyword sample order...")

    with h5py.File(subbiomes_path, "r") as subf, \
         h5py.File(keywords_path, "r") as keyf, \
         h5py.File(aligned_output, "w") as outf:

        sub_ids = decode_array(subf["sample_ids"][:])
        key_ids = decode_array(keyf["sample_ids"][:])

        sub_index = {sid: i for i, sid in enumerate(sub_ids)}
        n_samples = len(key_ids)

        dt = h5py.string_dtype(encoding="utf-8")

        outf.create_dataset("sample_ids", shape=(n_samples,), dtype=dt)
        outf.create_dataset("texts", shape=(n_samples,), dtype=dt)
        outf.create_dataset("embeddings", shape=(n_samples, embedding_dim), dtype="f4")

        missing_count = 0
        total_start_time = time.time()

        for start in range(0, n_samples, batch_size):
            batch_start_time = time.time()
            end = min(start + batch_size, n_samples)
            batch_ids = key_ids[start:end]

            batch_texts = []
            batch_embeds = []

            for sid in batch_ids:
                if sid in sub_index:
                    idx = sub_index[sid]

                    text = subf["texts"][idx]
                    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)

                    embed = subf["embeddings"][idx]
                else:
                    text = "NA"
                    embed = np.full(embedding_dim, np.nan, dtype="f4")
                    missing_count += 1

                batch_texts.append(text)
                batch_embeds.append(embed)

            outf["sample_ids"][start:end] = np.array(batch_ids, dtype=object)
            outf["texts"][start:end] = np.array(batch_texts, dtype=object)
            outf["embeddings"][start:end, :] = np.vstack(batch_embeds)

            batch_elapsed = time.time() - batch_start_time
            print(f" → Aligned {end}/{n_samples} samples — last batch took {batch_elapsed:.2f} sec")

            if end % 100000 == 0:
                total_elapsed = (time.time() - total_start_time) / 60
                print(f" ⏱ Reached {end} samples — total elapsed time: {total_elapsed:.2f} min")

        total_elapsed = (time.time() - total_start_time) / 60
        print(f"✅ Alignment complete: {n_samples} total samples, {missing_count} missing in sub-biomes")
        print(f"✅ Saved aligned file to {aligned_output} in {total_elapsed:.2f} minutes")


# ===== STEP 2: AVERAGE ALIGNED SUB-BIOME + KEYWORD EMBEDDINGS =====
def average_subbiome_keyword_embeddings():
    if os.path.exists(combined_output):
        print(f"⚠️ Combined file already exists, skipping: {combined_output}")
        return

    print("🔁 Averaging aligned sub-biome and keyword embeddings...")

    with h5py.File(aligned_output, "r") as subf, \
         h5py.File(keywords_path, "r") as keyf:

        sub_ids = decode_array(subf["sample_ids"][:])
        key_ids = decode_array(keyf["sample_ids"][:])

        if len(sub_ids) != len(key_ids):
            raise ValueError("Sub-biome and keyword files have different numbers of samples.")

        if not np.array_equal(sub_ids, key_ids):
            raise ValueError("Sample IDs are not in the same order. Aborting to avoid misalignment.")

        n_samples = len(key_ids)
        dt = h5py.string_dtype(encoding="utf-8")

        with h5py.File(combined_output, "w") as outf:
            outf.create_dataset("sample_ids", shape=(n_samples,), dtype=dt)
            outf.create_dataset("sub_texts", shape=(n_samples,), dtype=dt)
            outf.create_dataset("key_texts", shape=(n_samples,), dtype=dt)
            outf.create_dataset("embeddings", shape=(n_samples, embedding_dim), dtype="f4")

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                batch_ids = key_ids[start:end]

                sub_texts = subf["texts"][start:end]
                key_texts = keyf["texts"][start:end]

                sub_texts = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in sub_texts
                ]

                key_texts = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in key_texts
                ]

                sub_embeds = subf["embeddings"][start:end]
                key_embeds = keyf["embeddings"][start:end]

                avg_embeds = np.where(
                    np.isnan(sub_embeds) & np.isnan(key_embeds),
                    np.nan,
                    np.where(
                        np.isnan(sub_embeds),
                        key_embeds,
                        np.where(
                            np.isnan(key_embeds),
                            sub_embeds,
                            (sub_embeds + key_embeds) / 2
                        )
                    )
                )

                outf["sample_ids"][start:end] = np.array(batch_ids, dtype=object)
                outf["sub_texts"][start:end] = np.array(sub_texts, dtype=object)
                outf["key_texts"][start:end] = np.array(key_texts, dtype=object)
                outf["embeddings"][start:end, :] = avg_embeds

                print(f" → Averaged {end}/{n_samples} samples")

    print(f"✅ Saved averaged embeddings to {combined_output}")


# ===== RUN =====
check_input_file(subbiomes_path, "Sub-biome")
check_input_file(keywords_path, "Keywords")

align_subbiomes_to_keywords()
average_subbiome_keyword_embeddings()

count_samples_in_h5(subbiomes_path)
count_samples_in_h5(aligned_output)
count_samples_in_h5(keywords_path)
count_samples_in_h5(combined_output)



