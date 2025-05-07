#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:36:35 2025

@author: danielagaio
"""




import os
import pandas as pd
import numpy as np
import h5py
import random
import umap
import plotly.express as px
import time

# --------- Load biome labels ---------
def load_biome_labels(biome_labels_path):
    biomes = {}
    with open(biome_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                biomes[parts[0]] = parts[1]
    return biomes

# --------- Load embeddings for selected sample IDs ---------
def load_embeddings_h5(filepath, biome_labels, selected_sample_ids):
    selected_set = set(selected_sample_ids)

    with h5py.File(filepath, 'r') as f:
        sample_ids_raw = f['sample_ids'][:]
        sample_ids = sample_ids_raw.astype(str)

        # Find indices of selected sample IDs
        selected_indices = [i for i, sid in enumerate(sample_ids) if sid in selected_set]

        # Only load selected embeddings
        embeddings = f['embeddings'][selected_indices, :]

        # Initialize text holders
        sub_texts = None
        keywords = None

        # Decide which text fields to use
        if 'sub_texts' in f and 'key_texts' in f:
            # dual: sub-biome + keywords
            sub_raw = f['sub_texts'][selected_indices]
            key_raw = f['key_texts'][selected_indices]
            sub_texts = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in sub_raw])
            keywords = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in key_raw])
        elif 'texts' in f:
            text_raw = f['texts'][selected_indices]
            decoded_texts = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in text_raw])

            if 'sub_biomes' in os.path.basename(filepath):
                # treat as sub-biome file
                sub_texts = decoded_texts
                keywords = np.array(['-'] * len(sub_texts))
            elif 'keywords' in os.path.basename(filepath):
                # treat as keywords file
                keywords = decoded_texts
                sub_texts = np.array(['-'] * len(keywords))
            else:
                raise ValueError(f"Cannot determine role of 'texts' in {filepath}")
        else:
            raise ValueError(f"No recognized text datasets found in {filepath}")

    selected_sample_ids = sample_ids[selected_indices]
    labels = [biome_labels.get(sid, 'unknown') for sid in selected_sample_ids]

    # Build DataFrame with both columns guaranteed
    df = pd.DataFrame({
        'sample_id': selected_sample_ids,
        'biome_label': labels,
        'sub-biome': sub_texts,
        'keywords': keywords
    })

    return df, embeddings




# --------- Paths ---------
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings")
biome_labels_path = os.path.join(os.path.expanduser('~'), 'cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt')

subbiomes_path = os.path.join(work_dir, 'GPT_sub_biomes_embeddings.h5')
keywords_path = os.path.join(work_dir, 'GPT_keywords_embeddings.h5')
sb_keywords_path = os.path.join(work_dir, 'GPT_sub_biomes_keywords_embeddings.h5')

common_ids_path = os.path.join(work_dir, 'common_sample_ids_of_embeddings.npy')

# --------- Parameters ---------
samples_per_biome = 500  # adjust as needed
random_seed = 42

# --------- Load common sample IDs ---------
print(f"Loading common sample IDs from {common_ids_path}...")
common_sample_ids = np.load(common_ids_path)
print(f"Loaded {len(common_sample_ids)} common sample IDs.")

# --------- Load biome labels ---------
biome_labels = load_biome_labels(biome_labels_path)

# --------- Subsample sample IDs per biome ---------
print("Subsampling sample IDs per biome...")
random.seed(random_seed)
np.random.seed(random_seed)

# Build biome → sample_id list
biome_to_ids = {}
for sid in common_sample_ids:
    biome = biome_labels.get(sid, 'unknown')
    if biome not in biome_to_ids:
        biome_to_ids[biome] = []
    biome_to_ids[biome].append(sid)

# Select N random samples per biome
final_selected_ids = []
for biome, ids in biome_to_ids.items():
    if len(ids) <= samples_per_biome:
        chosen = ids
    else:
        chosen = random.sample(ids, samples_per_biome)
    final_selected_ids.extend(chosen)

print(f"Final selected samples: {len(final_selected_ids)}")

# --------- Load filtered embeddings ---------
print("Loading filtered embeddings...")
df_subbiomes, X_subbiomes = load_embeddings_h5(subbiomes_path, biome_labels, final_selected_ids)
df_keywords, X_keywords = load_embeddings_h5(keywords_path, biome_labels, final_selected_ids)
df_sb_keywords, X_sb_keywords = load_embeddings_h5(sb_keywords_path, biome_labels, final_selected_ids)

print("Embeddings loaded and ready for clustering.")









# --------- Helper: Run UMAP and add to DataFrame ---------
def run_umap(X, df, label_column='sub-biome'):
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)
    df['UMAP1'] = X_umap[:, 0]
    df['UMAP2'] = X_umap[:, 1]
    return df

# --------- Helper: Make interactive scatter plot ---------
def make_scatter_plot(df, title, hover_fields, biome_colors, out_path):
    fig = px.scatter(
        df,
        x='UMAP1', y='UMAP2',
        color='biome_label',
        color_discrete_map=biome_colors,
        hover_data=hover_fields,
        title=title,
        height=800,
        width=1000
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(legend_title_text='Biome Label')
    fig.write_html(out_path)
    print(f"Saved plot to {out_path}")
    return fig

# --------- Define biome colors ---------
biome_colors = {
    'water': '#8CC8CF',
    'plant': '#C0D184',
    'animal': '#C67D7B',
    'other': '#CCCCCC',
    'soil': '#CBBF82'
}

# --------- Run UMAP reductions ---------
start = time.time()
print("Running UMAP on sub-biomes...")
df_subbiomes = run_umap(X_subbiomes, df_subbiomes)
print(f"Done in {time.time() - start:.2f} seconds.")

start = time.time()
print("Running UMAP on keywords...")
df_keywords = run_umap(X_keywords, df_keywords)
print(f"Done in {time.time() - start:.2f} seconds.")

start = time.time()
print("Running UMAP on subbiomes + keywords...")
df_sb_keywords = run_umap(X_sb_keywords, df_sb_keywords)
print(f"Done in {time.time() - start:.2f} seconds.")


# --------- Define output paths ---------
out1 = os.path.join(work_dir, 'umap_subbiomes.html')
out2 = os.path.join(work_dir, 'umap_keywords.html')
out3 = os.path.join(work_dir, 'umap_sb_keywords.html')

# --------- Create and save plots ---------
fig1 = make_scatter_plot(
    df_subbiomes,
    title="Sub-biome embeddings",
    hover_fields={'sample_id': True, 'sub-biome': True, 'UMAP1': False, 'UMAP2': False},
    biome_colors=biome_colors,
    out_path=out1
)

fig2 = make_scatter_plot(
    df_keywords,
    title="Keyword embeddings",
    hover_fields={'sample_id': True, 'keywords': True, 'UMAP1': False, 'UMAP2': False},
    biome_colors=biome_colors,
    out_path=out2
)

fig3 = make_scatter_plot(
    df_sb_keywords,
    title="Subbiomes + keyword (avg) embeddings",
    hover_fields={'sample_id': True, 'sub-biome': True, 'keywords': True, 'UMAP1': False, 'UMAP2': False},
    biome_colors=biome_colors,
    out_path=out3
)

# --------- Optional: Show plots interactively ---------
fig1.show()
fig2.show()
fig3.show()

# --------- Print sample counts ---------
print(f"Subbiomes samples: {len(df_subbiomes)}")
print(f"Keywords samples: {len(df_keywords)}")
print(f"Subbiomes + keywords samples: {len(df_sb_keywords)}")














