#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create embeddings for GPT sub-biomes, GPT keywords, and optionally gold sub-biomes.

Examples:

python ~/github/metadata_mining/scripts/production/make_embeddings.py \
  --work_dir ~/MicrobeAtlasProject2024/production \
  --api_key_path ~/Desktop/keys/my_api_key_embeddings \
  --model text-embedding-3-small \
  --embedding_dim 1536

python ~/github/metadata_mining/scripts/production/make_embeddings.py \
  --work_dir ~/MicrobeAtlasProject2024/production \
  --api_key_path ~/Desktop/keys/my_api_key_embeddings \
  --model text-embedding-3-small \
  --embedding_dim 1536 \
  --include_gold \
  --gold_dict ~/MicrobeAtlasProject2024/gold_dict.pkl \
  --gold_subbiome_index 0
"""

import os
import time
import json
import pickle
import itertools
import argparse

import h5py
import openai
import numpy as np


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--work_dir",
    default=os.path.join(os.path.expanduser("~"), "MicrobeAtlasProject2024/production")
)
parser.add_argument(
    "--api_key_path",
    default=os.path.join(os.path.expanduser("~"), "Desktop/keys/my_api_key_embeddings")
)

parser.add_argument("--model", default="text-embedding-3-large")
parser.add_argument("--embedding_dim", type=int, default=3072)

parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--file_slice_size", type=int, default=10000)
parser.add_argument("--max_requests_per_round", type=int, default=100)
parser.add_argument("--wait_time", type=int, default=60)

parser.add_argument("--sub_biomes_input", default="GPT_sub_biomes.txt")
parser.add_argument("--sub_biomes_output", default=None)
parser.add_argument("--sub_biomes_state", default=None)

parser.add_argument("--keywords_input", default="GPT_keywords.txt")
parser.add_argument("--keywords_output", default=None)
parser.add_argument("--keywords_state", default=None)

# Gold sub-biome options
parser.add_argument("--include_gold", action="store_true")
parser.add_argument("--gold_dict", default=None)
parser.add_argument("--gold_subbiomes_output_txt", default="gold_sub_biomes.txt")
parser.add_argument("--gold_subbiomes_output_h5", default=None)
parser.add_argument("--gold_subbiomes_state", default=None)
parser.add_argument("--gold_subbiome_index", type=int, default=0)

args = parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

work_dir = os.path.expanduser(args.work_dir)
api_key_path = os.path.expanduser(args.api_key_path)

model = args.model
embedding_dim = args.embedding_dim
batch_size = args.batch_size
file_slice_size = args.file_slice_size
max_requests_per_round = args.max_requests_per_round
wait_time = args.wait_time

embeddings_dir = os.path.join(work_dir, "embeddings")
os.makedirs(embeddings_dir, exist_ok=True)


sub_output = (
    args.sub_biomes_output
    if args.sub_biomes_output
    else f"embeddings/GPT_sub_biomes_embeddings_{embedding_dim}.h5"
)

key_output = (
    args.keywords_output
    if args.keywords_output
    else f"embeddings/GPT_keywords_embeddings_{embedding_dim}.h5"
)

sub_state = (
    args.sub_biomes_state
    if args.sub_biomes_state
    else f"state_file_sub_biomes_{embedding_dim}.txt"
)

key_state = (
    args.keywords_state
    if args.keywords_state
    else f"state_file_keywords_{embedding_dim}.txt"
)


# =============================================================================
# OpenAI client
# =============================================================================

with open(api_key_path, "r") as f:
    openai_client = openai.OpenAI(api_key=f.read().strip())


# =============================================================================
# Helper functions
# =============================================================================

def resolve_work_path(path):
    """Resolve a path relative to work_dir unless already absolute."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(work_dir, path)


def write_gold_subbiomes_txt(gold_dict_path, output_txt_path, subbiome_index=0):
    """
    Convert gold_dict.pkl into a two-column file:
        sample_id <TAB> gold_sub_biome
    """
    with open(gold_dict_path, "rb") as f:
        gold_dict = pickle.load(f)

    n_written = 0
    n_skipped = 0

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for sample_id, value in gold_dict.items():

            if isinstance(value, (list, tuple)):
                if len(value) <= subbiome_index:
                    n_skipped += 1
                    continue
                subbiome = value[subbiome_index]
            elif isinstance(value, dict):
                subbiome = (
                    value.get("sub-biome")
                    or value.get("sub_biome")
                    or value.get("subbiome")
                    or value.get("gold_sub_biome")
                )
            else:
                subbiome = value

            if subbiome is None:
                n_skipped += 1
                continue

            subbiome = str(subbiome).strip()
            if not subbiome or subbiome.lower() in {"nan", "none", "na", "n/a"}:
                n_skipped += 1
                continue

            out.write(f"{sample_id}\t{subbiome}\n")
            n_written += 1

    print(f"✅ Wrote {n_written} gold sub-biomes to {output_txt_path}")
    if n_skipped:
        print(f"⚠️ Skipped {n_skipped} gold entries without usable sub-biome labels")


def read_samples_slice(file_path, start, end, keywords=False):
    """
    Read a slice from a two-column sample/text file.
    Expected format:
        sample_id <TAB> text
    """
    samples = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in itertools.islice(f, start, end):
            line = line.rstrip("\n")
            parts = line.split("\t", 1)

            if len(parts) != 2:
                continue

            sample_id, text = parts
            sample_id = sample_id.strip()
            text = text.strip()

            if not sample_id or not text:
                continue

            if keywords:
                text = text.strip("{}").replace(",", " ")

            if text.lower() in {"nan", "none", "na", "n/a"}:
                continue

            samples[sample_id] = text

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
                model=model
            )

            embeddings = [item.embedding for item in response.data]

            for j, sample_id in enumerate(sample_ids_chunk):
                embeddings_dict[sample_id] = {
                    "embedding": embeddings[j],
                    "text": samples_dict[sample_id]
                }

            request_count += 1
            elapsed = time.time() - start_time

            print(
                f" → Request {request_count}: "
                f"{len(sample_ids_chunk)} samples in {elapsed:.2f}s"
            )

            if request_count % max_requests_per_round == 0:
                print(f" → Reached {request_count} requests, waiting {wait_time}s...")
                time.sleep(wait_time)

        except Exception as e:
            print(
                f"❌ Failed embedding batch "
                f"{sample_ids_chunk[0]}–{sample_ids_chunk[-1]}: {e}"
            )

    return embeddings_dict


def update_state_file(state_file, index):
    with open(state_file, "w") as f:
        json.dump({"last_sample": index}, f)


def get_current_index(state_file):
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
            return state.get("last_sample", 0)
    return 0


def append_embeddings_to_h5(output_path, ids, texts, emb_array):
    """
    Append sample IDs, texts, and embeddings to an HDF5 file.
    """
    dt = h5py.string_dtype(encoding="utf-8")

    if os.path.exists(output_path):
        with h5py.File(output_path, "r+") as h5f:

            if "sample_ids" not in h5f:
                h5f.create_dataset(
                    "sample_ids",
                    data=ids,
                    maxshape=(None,),
                    dtype=dt
                )
            else:
                old_n = h5f["sample_ids"].shape[0]
                h5f["sample_ids"].resize(old_n + len(ids), axis=0)
                h5f["sample_ids"][old_n:] = ids

            if "texts" not in h5f:
                h5f.create_dataset(
                    "texts",
                    data=texts,
                    maxshape=(None,),
                    dtype=dt
                )
            else:
                old_n = h5f["texts"].shape[0]
                h5f["texts"].resize(old_n + len(texts), axis=0)
                h5f["texts"][old_n:] = texts

            if "embeddings" not in h5f:
                h5f.create_dataset(
                    "embeddings",
                    data=emb_array,
                    maxshape=(None, embedding_dim),
                    dtype="f4"
                )
            else:
                old_n = h5f["embeddings"].shape[0]
                h5f["embeddings"].resize(old_n + len(ids), axis=0)
                h5f["embeddings"][old_n:] = emb_array

    else:
        with h5py.File(output_path, "w") as h5f:
            h5f.create_dataset(
                "sample_ids",
                data=ids,
                maxshape=(None,),
                dtype=dt
            )
            h5f.create_dataset(
                "texts",
                data=texts,
                maxshape=(None,),
                dtype=dt
            )
            h5f.create_dataset(
                "embeddings",
                data=emb_array,
                maxshape=(None, embedding_dim),
                dtype="f4"
            )


def process_file(input_file, output_file, state_file, keywords=False):
    input_path = resolve_work_path(input_file)
    output_path = resolve_work_path(output_file)
    state_path = os.path.join(embeddings_dir, state_file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)

    if not os.path.exists(input_path):
        print(f"⚠️ Input file not found, skipping: {input_path}")
        return

    start_idx = get_current_index(state_path)
    total_processed = 0
    total_start_time = time.time()

    print("\n" + "=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"State:  {state_path}")
    print("=" * 80)

    while True:
        slice_start = start_idx
        slice_end = slice_start + file_slice_size

        samples = read_samples_slice(
            input_path,
            slice_start,
            slice_end,
            keywords=keywords
        )

        if not samples:
            print(f"✅ All samples processed for {os.path.basename(input_path)}")
            break

        print(
            f"Processing lines {slice_start}–{slice_end - 1} "
            f"({len(samples)} usable samples)"
        )

        slice_start_time = time.time()
        embeddings = get_embeddings(samples)

        if not embeddings:
            print("⚠️ No embeddings generated for this slice; stopping this file")
            break

        ids = list(embeddings.keys())
        texts = [embeddings[s]["text"] for s in ids]
        emb_array = np.asarray(
            [embeddings[s]["embedding"] for s in ids],
            dtype=np.float32
        )

        if emb_array.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {embedding_dim}, "
                f"got {emb_array.shape[1]}. Check --model and --embedding_dim."
            )

        append_embeddings_to_h5(output_path, ids, texts, emb_array)

        slice_elapsed = time.time() - slice_start_time
        print(f" → Slice done in {slice_elapsed / 60:.2f} min")

        start_idx = slice_end
        total_processed += len(ids)

        update_state_file(state_path, start_idx)

    total_elapsed = time.time() - total_start_time
    print(
        f"✅ Finished {os.path.basename(input_path)}: "
        f"{total_processed} samples in {total_elapsed / 60:.2f} min"
    )


# =============================================================================
# Build list of files to embed
# =============================================================================

input_files = [
    {
        "input": args.sub_biomes_input,
        "output": sub_output,
        "state": sub_state,
        "keywords": False,
        "label": "GPT sub-biomes",
    },
    {
        "input": args.keywords_input,
        "output": key_output,
        "state": key_state,
        "keywords": True,
        "label": "GPT keywords",
    },
]


if args.include_gold:
    if args.gold_dict is None:
        raise ValueError("--include_gold requires --gold_dict")

    gold_dict_path = os.path.expanduser(args.gold_dict)
    gold_txt_path = os.path.join(work_dir, args.gold_subbiomes_output_txt)

    write_gold_subbiomes_txt(
        gold_dict_path=gold_dict_path,
        output_txt_path=gold_txt_path,
        subbiome_index=args.gold_subbiome_index
    )

    gold_h5_output = (
        args.gold_subbiomes_output_h5
        if args.gold_subbiomes_output_h5
        else f"embeddings/gold_sub_biomes_embeddings_{embedding_dim}.h5"
    )

    gold_state = (
        args.gold_subbiomes_state
        if args.gold_subbiomes_state
        else f"state_file_gold_sub_biomes_{embedding_dim}.txt"
    )

    input_files.append(
        {
            "input": args.gold_subbiomes_output_txt,
            "output": gold_h5_output,
            "state": gold_state,
            "keywords": False,
            "label": "gold sub-biomes",
        }
    )


# =============================================================================
# Run
# =============================================================================

overall_start_time = time.time()

print("\nEmbedding run configuration")
print("---------------------------")
print(f"work_dir:       {work_dir}")
print(f"model:          {model}")
print(f"embedding_dim:  {embedding_dim}")
print(f"batch_size:     {batch_size}")
print(f"slice_size:     {file_slice_size}")
print(f"include_gold:   {args.include_gold}")

for item in input_files:
    print(f"\n🚀 Starting: {item['label']}")
    process_file(
        input_file=item["input"],
        output_file=item["output"],
        state_file=item["state"],
        keywords=item["keywords"]
    )
    print(f"✅ Completed: {item['label']}")

overall_elapsed = time.time() - overall_start_time
print(f"\n🏁 All embedding runs completed in {overall_elapsed / 60:.2f} minutes")




# before adding gold dict embeddings: 
# =============================================================================
# # run as: 
# 
# # python ~/github/metadata_mining/scripts/production/make_embeddings.py \
# #   --work_dir ~/MicrobeAtlasProject2024/production \
# #   --model text-embedding-3-small \
# #   --embedding_dim 1536
#   
# # python ~/github/metadata_mining/scripts/production/make_embeddings.py \
# #   --work_dir ~/MicrobeAtlasProject2024/production \
# #   --model text-embedding-3-large \
# #   --embedding_dim 3072
# 
# 
# import os
# import openai
# import h5py
# import numpy as np
# import time
# import json
# import itertools
# import argparse
# 
# 
# # ===== ARGUMENTS =====
# parser = argparse.ArgumentParser()
# 
# parser.add_argument("--work_dir", default=os.path.join(os.path.expanduser("~"), "MicrobeAtlasProject2024/production"))
# parser.add_argument("--api_key_path", default=os.path.join(os.path.expanduser("~"), "Desktop/keys/my_api_key_embeddings"))
# 
# parser.add_argument("--model", default="text-embedding-3-large")
# parser.add_argument("--embedding_dim", type=int, default=3072)
# 
# parser.add_argument("--batch_size", type=int, default=1000)
# parser.add_argument("--file_slice_size", type=int, default=10000)
# parser.add_argument("--max_requests_per_round", type=int, default=100)
# parser.add_argument("--wait_time", type=int, default=60)
# 
# parser.add_argument("--sub_biomes_input", default="GPT_sub_biomes.txt")
# parser.add_argument("--sub_biomes_output", default=None)
# parser.add_argument("--sub_biomes_state", default=None)
# 
# parser.add_argument("--keywords_input", default="GPT_keywords.txt")
# parser.add_argument("--keywords_output", default=None)
# parser.add_argument("--keywords_state", default=None)
# 
# args = parser.parse_args()
# 
# 
# # ===== CONFIGURATION =====
# work_dir = args.work_dir
# api_key_path = args.api_key_path
# 
# sub_output = (
#     args.sub_biomes_output
#     if args.sub_biomes_output
#     else f"embeddings/GPT_sub_biomes_embeddings_{args.embedding_dim}.h5"
# )
# 
# key_output = (
#     args.keywords_output
#     if args.keywords_output
#     else f"embeddings/GPT_keywords_embeddings_{args.embedding_dim}.h5"
# )
# 
# sub_state = (
#     args.sub_biomes_state
#     if args.sub_biomes_state
#     else f"state_file_sub_biomes_{args.embedding_dim}.txt"
# )
# 
# key_state = (
#     args.keywords_state
#     if args.keywords_state
#     else f"state_file_keywords_{args.embedding_dim}.txt"
# )
# 
# input_files = [
#     (args.sub_biomes_input, sub_output, sub_state, False),
#     (args.keywords_input, key_output, key_state, True)
# ]
# 
# batch_size = args.batch_size
# file_slice_size = args.file_slice_size
# max_requests_per_round = args.max_requests_per_round
# wait_time = args.wait_time
# embedding_dim = args.embedding_dim
# model = args.model
# 
# # ===== SETUP API =====
# with open(api_key_path, 'r') as f:
#     openai_client = openai.OpenAI(api_key=f.read().strip())
# 
# # ===== HELPER FUNCTIONS =====
# def read_samples_slice(file_path, start, end, keywords=False):
#     samples = {}
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in itertools.islice(f, start, end):
#             parts = line.strip().split('\t')
#             if len(parts) != 2:
#                 continue
#             sample_id, text = parts
#             if keywords:
#                 text = text.strip('{}').replace(',', ' ')
#             samples[sample_id] = text.strip()
#     return samples
# 
# def get_embeddings(samples_dict):
#     embeddings_dict = {}
#     sample_ids = list(samples_dict.keys())
#     descriptions = list(samples_dict.values())
# 
#     request_count = 0
#     for i in range(0, len(descriptions), batch_size):
#         chunk = descriptions[i:i + batch_size]
#         sample_ids_chunk = sample_ids[i:i + batch_size]
#         start_time = time.time()
#         try:
#             response = openai_client.embeddings.create(
#                 input=chunk,
#                 model=model
#             )
#             embeddings = [item.embedding for item in response.data]
#             for j, sample_id in enumerate(sample_ids_chunk):
#                 embeddings_dict[sample_id] = {
#                     'embedding': embeddings[j],
#                     'text': samples_dict[sample_id]
#                 }
#             request_count += 1
#             elapsed = time.time() - start_time
#             print(f" → Batch {request_count} ({len(sample_ids_chunk)} samples) took {elapsed:.2f}s")
#             if request_count % max_requests_per_round == 0:
#                 print(f" → Reached {request_count} requests, waiting {wait_time}s...")
#                 time.sleep(wait_time)
#         except Exception as e:
#             print(f"Failed batch {sample_ids_chunk[0]}–{sample_ids_chunk[-1]}: {e}")
#     return embeddings_dict
# 
# def update_state_file(state_file, index):
#     with open(state_file, 'w') as f:
#         json.dump({'last_sample': index}, f)
# 
# def get_current_index(state_file):
#     if os.path.exists(state_file):
#         with open(state_file, 'r') as f:
#             state = json.load(f)
#             return state.get('last_sample', 0)
#     return 0
# 
# def process_file(input_file, output_file, state_file, keywords):
#     input_path = os.path.join(work_dir, input_file)
#     output_path = os.path.join(work_dir, output_file)
#     state_path = os.path.join(work_dir, 'embeddings', state_file)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     os.makedirs(os.path.dirname(state_path), exist_ok=True)
# 
#     start_idx = get_current_index(state_path)
#     total_processed = 0
#     total_start_time = time.time()
# 
#     while True:
#         slice_start = start_idx
#         slice_end = slice_start + file_slice_size
#         samples = read_samples_slice(input_path, slice_start, slice_end, keywords)
#         if not samples:
#             print(f"✅ All samples processed for {input_file}")
#             break
#         print(f"Processing samples {slice_start}–{slice_end -1} ({len(samples)})")
# 
#         slice_start_time = time.time()
#         embeddings = get_embeddings(samples)
#         if not embeddings:
#             print("⚠️ No embeddings generated, skipping slice")
#             break
# 
#         ids = list(embeddings.keys())
#         texts = [embeddings[s]['text'] for s in ids]
#         emb_array = np.vstack([embeddings[s]['embedding'] for s in ids])
#         dt = h5py.string_dtype(encoding='utf-8')
# 
#         if os.path.exists(output_path):
#             with h5py.File(output_path, 'r+') as h5f:
#                 for name, data, shape in [('sample_ids', ids, (None,)), ('texts', texts, (None,)), ('embeddings', emb_array, (None, embedding_dim))]:
#                     if name not in h5f:
#                         maxshape = (None,) if name != 'embeddings' else (None, embedding_dim)
#                         dtype = dt if name != 'embeddings' else 'f4'
#                         h5f.create_dataset(name, data=data, maxshape=maxshape, dtype=dtype)
#                     else:
#                         h5f[name].resize(h5f[name].shape[0] + len(ids), axis=0)
#                         h5f[name][-len(ids):] = data
#         else:
#             with h5py.File(output_path, 'w') as h5f:
#                 h5f.create_dataset('sample_ids', data=ids, maxshape=(None,), dtype=dt)
#                 h5f.create_dataset('texts', data=texts, maxshape=(None,), dtype=dt)
#                 h5f.create_dataset('embeddings', data=emb_array, maxshape=(None, embedding_dim), dtype='f4')
#         
#         slice_elapsed = time.time() - slice_start_time
#         print(f" → Slice done in {slice_elapsed/60:.2f} min")
# 
#         start_idx = slice_end
#         total_processed += len(ids)
#         update_state_file(state_path, start_idx)
#     
#     total_elapsed = time.time() - total_start_time
#     print(f"✅ Finished {input_file}: {total_processed} samples in {total_elapsed/60:.2f} min")
# 
# # ===== RUN PIPELINE =====
# overall_start_time = time.time()
# 
# for infile, outfile, statefile, keywords in input_files:
#     process_file(infile, outfile, statefile, keywords)
#     print(f"✅ Completed {infile}\n")
# 
# overall_elapsed = time.time() - overall_start_time
# print(f"🏁 All embedding runs completed in {overall_elapsed/60:.2f} minutes")
# =============================================================================




