#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:36:35 2025

@author: danielagaio
"""


import os
import pandas as pd
import numpy as np
import umap
import plotly.express as px
import h5py

# Helper function to load embeddings from HDF5
def load_embeddings_h5(filepath, biome_labels_path, limit_samples=None):
    # Load biome labels
    biomes = {}
    with open(biome_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                biomes[parts[0]] = parts[1]

    # Load HDF5 data
    with h5py.File(filepath, 'r') as f:
        total_samples = f['sample_ids'].shape[0]
        sample_ids_raw = f['sample_ids'][:]
        texts_raw = f['texts'][:]
        vectors = f['embeddings'][:]

    # Proper UTF-8 decoding
    sample_ids = sample_ids_raw.astype(str)
    texts = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in texts_raw])

    # Map biome labels
    labels = [biomes.get(sid, 'unknown') for sid in sample_ids]

    # Build DataFrame
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'biome_label': labels,
        'sub-biome': texts
    })

    # Add embedding vectors to DataFrame (for easier selection)
    df['vector'] = list(vectors)

    # Subsample equally across biomes
    if limit_samples is not None:
        biomes_in_df = df['biome_label'].unique()
        samples_per_biome = limit_samples // len(biomes_in_df)
        df = df.groupby('biome_label', group_keys=False).apply(
            lambda x: x.sample(min(samples_per_biome, len(x)), random_state=42)
        )

    # Extract embedding matrix
    X = np.vstack(df['vector'].to_numpy())

    # Drop the 'vector' column
    df = df.drop(columns=['vector'])

    return df, X




# Paths
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings")
biome_labels_path = os.path.join(os.path.expanduser('~'), 'cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt')

subbiomes_path = os.path.join(work_dir, 'GPT_sub_biomes_embeddings.h5')
keywords_path = os.path.join(work_dir, 'GPT_keywords_embeddings.h5')
sb_keywords_path = os.path.join(work_dir, 'GPT_sub_biomes_keywords_embeddings.h5')


# Load embeddings
df_subbiomes, X_subbiomes = load_embeddings_h5(subbiomes_path, biome_labels_path, limit_samples=6000)
df_keywords, X_keywords = load_embeddings_h5(keywords_path, biome_labels_path, limit_samples=6000)
df_sb_keywords, X_sb_keywords = load_embeddings_h5(sb_keywords_path, biome_labels_path, limit_samples=6000)



# UMAP Reduction
reducer = umap.UMAP(random_state=42)
X_subbiomes_umap = reducer.fit_transform(X_subbiomes)

reducer2 = umap.UMAP(random_state=42)
X_keywords_umap = reducer2.fit_transform(X_keywords)

reducer3 = umap.UMAP(random_state=42)
X_sb_keywords_umap = reducer3.fit_transform(X_sb_keywords)


# Add UMAP coords
df_subbiomes['UMAP1'] = X_subbiomes_umap[:, 0]
df_subbiomes['UMAP2'] = X_subbiomes_umap[:, 1]

df_keywords['UMAP1'] = X_keywords_umap[:, 0]
df_keywords['UMAP2'] = X_keywords_umap[:, 1]

df_sb_keywords['UMAP1'] = X_sb_keywords_umap[:, 0]
df_sb_keywords['UMAP2'] = X_sb_keywords_umap[:, 1]


df_keywords = df_keywords.rename(columns={'sub-biome': 'keywords'})
df_sb_keywords = df_sb_keywords.rename(columns={'sub-biome': 'keywords'})


# Define biome colors
biome_colors = {
    'water': '#8CC8CF',
    'plant': '#C0D184',
    'animal': '#C67D7B',
    'other': '#CCCCCC',
    'soil': '#CBBF82'
}

# Interactive Plot: Sub-biomes
fig1 = px.scatter(
    df_subbiomes,
    x='UMAP1', y='UMAP2',
    color='biome_label',
    color_discrete_map=biome_colors,
    hover_data={
    'sample_id': True,
    'sub-biome': True,   
    'UMAP1': False,
    'UMAP2': False},
    title="Sub-biome embeddings",
    height=800,
    width=1000
)
fig1.update_traces(marker=dict(size=7, opacity=0.7))
fig1.update_layout(legend_title_text='Biome Label')

# Interactive Plot: Keywords
fig2 = px.scatter(
    df_keywords,
    x='UMAP1', y='UMAP2',
    color='biome_label',
    color_discrete_map=biome_colors,
    hover_data={
        'sample_id': True,
        'keywords': True, 
        'UMAP1': False,
        'UMAP2': False,
    },
    title="Keyword embeddings",
    height=800,
    width=1000
)
fig2.update_traces(marker=dict(size=7, opacity=0.7))
fig2.update_layout(legend_title_text='Biome Label')



# Interactive Plot: sb_keywords
fig3 = px.scatter(
    df_sb_keywords,
    x='UMAP1', y='UMAP2',
    color='biome_label',
    color_discrete_map=biome_colors,
    hover_data={
        'sample_id': True,
        'sub-biome': True, 
        'keywords': True, 
        'UMAP1': False,
        'UMAP2': False,
    },
    title="Subbiomes+keyword (avg) embeddings",
    height=800,
    width=1000
)
fig3.update_traces(marker=dict(size=7, opacity=0.7))
fig3.update_layout(legend_title_text='Biome Label')



# Show both
fig1.show()
fig2.show()
fig3.show()

# Optional: Save HTMLs
out1 = os.path.join(work_dir, 'umap_subbiomes.html')
out2 = os.path.join(work_dir, 'umap_keywords.html')
out3 = os.path.join(work_dir, 'umap_sb_keywords.html')
fig1.write_html(out1)
fig2.write_html(out2)
fig3.write_html(out3)




print("Subbiomes samples:", len(df_subbiomes))
print("Keywords samples:", len(df_keywords))
print("Merged samples:", len(df_sb_keywords))






