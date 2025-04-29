#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:36:35 2025

@author: danielagaio
"""



import json
import pandas as pd
import numpy as np
import umap
import plotly.express as px

# Helper function to load embeddings
def load_embeddings(filepath, biome_labels_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)

    biomes = {}
    with open(biome_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                biomes[parts[0]] = parts[1]

    sample_ids = []
    vectors = []
    labels = []
    texts = []

    for sample_id, info in embeddings_data.items():
        sample_ids.append(sample_id)
        vectors.append(info['embedding'])
        labels.append(biomes.get(sample_id, 'unknown'))
        texts.append(info['text'])

    df = pd.DataFrame({
        'sample_id': sample_ids,
        'biome_label': labels,
        'sub-biome': texts
    })

    X = np.array(vectors)
    return df, X

# Paths
work_dir = os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings")
biome_labels_path = 'cloudstor/Gaio/MicrobeAtlasProject/Hackathon/GPT_biomes.txt'

subbiomes_path = os.path.join(work_dir, 'GPT_sub_biomes_embeddings.json')
keywords_path = os.path.join(work_dir, 'GPT_keywords_embeddings.json')

# Load embeddings
df_subbiomes, X_subbiomes = load_embeddings(subbiomes_path, biome_labels_path)
df_keywords, X_keywords = load_embeddings(keywords_path, biome_labels_path)

# UMAP Reduction
reducer = umap.UMAP(random_state=42)
X_subbiomes_umap = reducer.fit_transform(X_subbiomes)

reducer2 = umap.UMAP(random_state=42)
X_keywords_umap = reducer2.fit_transform(X_keywords)

# Add UMAP coords
df_subbiomes['UMAP1'] = X_subbiomes_umap[:,0]
df_subbiomes['UMAP2'] = X_subbiomes_umap[:,1]

df_keywords['UMAP1'] = X_keywords_umap[:,0]
df_keywords['UMAP2'] = X_keywords_umap[:,1]

# Interactive Plot: Sub-biomes
fig1 = px.scatter(
    df_subbiomes,
    x='UMAP1', y='UMAP2',
    color='biome_label',
    hover_data={
        'sample_id': True,
        'sub-biome': True,
        'UMAP1': False,
        'UMAP2': False,
    },
    title="UMAP Projection of Sub-biome Embeddings",
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
    hover_data={
        'sample_id': True,
        'sub-biome': True,
        'UMAP1': False,
        'UMAP2': False,
    },
    title="UMAP Projection of Keyword Embeddings",
    height=800,
    width=1000
)
fig2.update_traces(marker=dict(size=7, opacity=0.7))
fig2.update_layout(legend_title_text='Biome Label')

# Show both
fig1.show()
fig2.show()

# Optional: Save HTMLs
out1=os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings/umap_subbiomes.html")
out2=os.path.join(os.path.expanduser('~'), "cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings/umap_keywords.html")
fig1.write_html(os.path.join(work_dir, out1))
fig2.write_html(os.path.join(work_dir, out2))


