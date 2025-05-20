#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:17:20 2025

@author: danielagaio
"""


import os
import json
import time
import numpy as np
import pandas as pd
import h5py
import umap
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer


# --------- Load biome labels ---------
def load_biome_labels(biome_labels_path):
    biomes = {}
    with open(biome_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                biomes[parts[0]] = parts[1]
    return biomes


# --------- Robust loading function ---------
def load_embeddings_h5(filepath, biome_labels, selected_sample_ids):
    selected_set = set(selected_sample_ids)

    with h5py.File(filepath, 'r') as f:
        sample_ids_raw = f['sample_ids'][:]
        sample_ids = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sample_ids_raw])
        selected_indices = [i for i, sid in enumerate(sample_ids) if sid in selected_set]

        embeddings = f['embeddings'][selected_indices, :]

        sub_texts, keywords = None, None

        if 'sub_texts' in f and 'key_texts' in f:
            sub_raw = f['sub_texts'][selected_indices]
            key_raw = f['key_texts'][selected_indices]
            sub_texts = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in sub_raw])
            keywords = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in key_raw])
        elif 'texts' in f:
            text_raw = f['texts'][selected_indices]
            decoded_texts = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in text_raw])
            if 'sub_biomes' in os.path.basename(filepath):
                sub_texts = decoded_texts
                keywords = np.array(['-'] * len(sub_texts))
            elif 'keywords' in os.path.basename(filepath):
                keywords = decoded_texts
                sub_texts = np.array(['-'] * len(keywords))
            else:
                raise ValueError(f"Cannot determine role of 'texts' in {filepath}")
        else:
            raise ValueError(f"No recognized text datasets found in {filepath}")

    selected_sample_ids = sample_ids[selected_indices]
    labels = [biome_labels.get(sid, 'unknown') for sid in selected_sample_ids]

    df = pd.DataFrame({
        'sample_id': selected_sample_ids,
        'biome_label': labels,
        'sub-biome': sub_texts,
        'keywords': keywords
    })

    return df, embeddings


# --------- Clustering helpers ---------
def filter_nan_rows(X, df):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], df.loc[mask].reset_index(drop=True)


def run_umap(X):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(X)


def cluster_umap(X_umap, eps=0.7, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    return clustering.fit_predict(X_umap)


def name_clusters(df, cluster_labels, use_column='sub-biome'):
    df['cluster'] = cluster_labels
    cluster_names = {}
    for cluster in set(cluster_labels):
        if cluster == -1:
            cluster_names[cluster] = 'trash'
            continue
        texts = df[df['cluster'] == cluster][use_column].tolist()
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        word_counts = np.asarray(X.sum(axis=0)).ravel()
        vocab = vectorizer.get_feature_names_out()
        if len(vocab) == 0:
            name = 'misc'
        else:
            name = vocab[np.argmax(word_counts)]
        cluster_names[cluster] = name
    df['cluster_name'] = df['cluster'].map(cluster_names)
    return df, cluster_names


def compute_centroids(X, cluster_labels):
    centroids = {}
    for cluster in set(cluster_labels):
        if cluster == -1:
            continue
        points = X[cluster_labels == cluster]
        center = points.mean(axis=0)
        radius = np.percentile(np.linalg.norm(points - center, axis=1), 95)
        centroids[cluster] = {'center': center.tolist(), 'radius': radius}
    return centroids


# --------- Main pipeline ---------
if __name__ == '__main__':
    base_dir = os.path.join(os.path.expanduser('~'), "Desktop/MicrobeAtlasProject/Hackathon")
    work_dir = os.path.join(base_dir, "embeddings")
    biome_labels_path = os.path.join(base_dir, 'GPT_biomes.txt')
    sampling_path = os.path.join(work_dir, "sampling", "sampling_nspb1000_seed22.txt")
    embedding_path = os.path.join(work_dir, 'GPT_sub_biomes_embeddings_aligned.h5')  # adjust as needed

    # Load sample IDs and labels
    with open(sampling_path, 'r') as f:
        selected_sample_ids = [line.strip() for line in f.readlines()]
    biome_labels = load_biome_labels(biome_labels_path)

    # Load data
    df, X = load_embeddings_h5(embedding_path, biome_labels, selected_sample_ids)
    X, df = filter_nan_rows(X, df)

    # Run UMAP
    X_umap = run_umap(X)

    # Cluster
    labels = cluster_umap(X_umap)

    # Name clusters (choose 'sub-biome' or 'keywords')
    df_named, cluster_names = name_clusters(df, labels, use_column='sub-biome')

    # Get centroids
    centroids = compute_centroids(X, labels)

    # Save results
    suffix = os.path.splitext(os.path.basename(sampling_path))[0].replace("sampling_", "")
    df_named.to_csv(os.path.join(work_dir, f'clusters_{suffix}.csv'), index=False)
    with open(os.path.join(work_dir, f'centroids_{suffix}.json'), 'w') as f:
        json.dump({str(k): v for k, v in centroids.items()}, f)
    with open(os.path.join(work_dir, f'cluster_names_{suffix}.json'), 'w') as f:
        json.dump({str(k): v for k, v in cluster_names.items()}, f)

    print("✅ Clustering complete. Outputs saved.")




import plotly.express as px

df_named['UMAP1'] = X_umap[:, 0]
df_named['UMAP2'] = X_umap[:, 1]


fig = px.scatter(
    df_named,
    x='UMAP1',
    y='UMAP2',
    color='cluster_name',
    hover_data={
        'sample_id': True,
        'biome_label': True,
        'sub-biome': True,
        'keywords': True,
        'cluster': True
    },
    title='UMAP of clustered sub-biome embeddings',
    height=800,
    width=1000
)

fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(legend_title_text='Cluster Name')
fig.show()



fig.write_html(os.path.join(work_dir, f'clusters_umap_{suffix}.html'))
print(f"✅ Saved plot to: clusters_umap_{suffix}.html")










import os
import h5py
import numpy as np
import pandas as pd
import faiss
from collections import Counter
import json

# --------- Config ---------
embedding_file = os.path.expanduser("~/Desktop/MicrobeAtlasProject/Hackathon/embeddings/GPT_sub_biomes_embeddings_aligned.h5")
clustered_csv = os.path.expanduser("~/Desktop/MicrobeAtlasProject/Hackathon/embeddings/clusters_nspb1000_seed22.csv")
output_csv = clustered_csv.replace(".csv", "_with_new_assignments.csv")
faiss_index_path = embedding_file.replace(".h5", "_faiss.index")

k_neighbors = 50
batch_size = 50000

# --------- Build FAISS index from HDF5 ---------
def build_faiss_index(h5_path, index_out_path, batch_size=50000):
    with h5py.File(h5_path, 'r') as f:
        total = f['embeddings'].shape[0]
        d = f['embeddings'].shape[1]

        index = faiss.IndexFlatL2(d)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = f['embeddings'][start:end]
            batch = np.asarray(batch)
            index.add(batch)
            print(f"✅ Added batch {start}–{end} to FAISS index")

        faiss.write_index(index, index_out_path)
        print(f"✅ FAISS index written to: {index_out_path}")

# --------- Load clustered samples ---------
def load_cluster_mapping(clustered_csv):
    df = pd.read_csv(clustered_csv)
    return df[['sample_id', 'cluster']].set_index('sample_id').to_dict()['cluster']

# --------- Map new embeddings to clusters ---------
def assign_clusters_to_new_samples(new_embeddings, sample_ids, faiss_index, id_lookup, k=50):
    _, indices = faiss_index.search(new_embeddings, k)
    assignments = []

    for neighbors in indices:
        neighbor_ids = [id_lookup[i] for i in neighbors if i in id_lookup]
        vote = Counter(neighbor_ids).most_common(1)
        assigned = vote[0][0] if vote else 'unclassified'
        assignments.append(assigned)

    return assignments

# --------- MAIN ---------
if __name__ == "__main__":
    # Step 1: Build FAISS index (only if not already created)
    if not os.path.exists(faiss_index_path):
        build_faiss_index(embedding_file, faiss_index_path, batch_size=batch_size)
    else:
        print(f"ℹ️ FAISS index already exists: {faiss_index_path}")

    # Step 2: Load clustered samples and cluster map
    cluster_df = pd.read_csv(clustered_csv)
    cluster_map = dict(zip(cluster_df['sample_id'], cluster_df['cluster']))
    known_sample_ids = list(cluster_df['sample_id'])

    # Step 3: Load full sample_ids from HDF5
    with h5py.File(embedding_file, 'r') as f:
        all_ids = [s.decode('utf-8') for s in f['sample_ids']]
        all_embeddings = f['embeddings'][:]

    # Step 4: Create reverse index for FAISS id → sample_id → cluster
    id_to_cluster = {i: cluster_map.get(sid, 'unclassified') for i, sid in enumerate(all_ids)}

    # Step 5: Load FAISS index
    index = faiss.read_index(faiss_index_path)

    # Step 6: Assign clusters to *all* samples
    print("🔍 Assigning clusters to all embeddings...")
    assignments = assign_clusters_to_new_samples(all_embeddings, all_ids, index, id_to_cluster, k=k_neighbors)

    # Step 7: Save output
    output_df = pd.DataFrame({
        'sample_id': all_ids,
        'assigned_cluster': assignments
    })
    output_df.to_csv(output_csv, index=False)
    print(f"✅ Assigned clusters written to: {output_csv}")


