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
import time
import umap
import plotly.express as px

# --------- Helper: Load biome labels ---------
def load_biome_labels(biome_labels_path):
    biomes = {}
    with open(biome_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                biomes[parts[0]] = parts[1]
    return biomes

# --------- Helper: Load embeddings from h5 ---------
def load_embeddings_h5(filepath, biome_labels, selected_sample_ids):
    selected_set = set(selected_sample_ids)

    with h5py.File(filepath, 'r') as f:
        sample_ids_raw = f['sample_ids'][:]
        sample_ids = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sample_ids_raw])

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
base_dir = os.path.join(os.path.expanduser('~'), "Desktop/MicrobeAtlasProject/Hackathon")
work_dir = os.path.join(base_dir, "embeddings")
sampling_dir = os.path.join(work_dir, "sampling")
biome_labels_path = os.path.join(base_dir, 'GPT_biomes.txt')

subbiomes_path = os.path.join(work_dir, 'GPT_sub_biomes_embeddings_aligned.h5')
keywords_path = os.path.join(work_dir, 'GPT_keywords_embeddings.h5')
sb_keywords_path = os.path.join(work_dir, 'GPT_sub_biomes_keywords_embeddings.h5')

# --------- List available sampling files ---------
sampling_files = [f for f in os.listdir(sampling_dir) if f.endswith('.txt')]
if not sampling_files:
    print("❌ No sampling files found in the sampling directory.")
    exit(1)

print("Available sampling files:")
for i, fname in enumerate(sampling_files, 1):
    print(f"{i}) {fname}")

choice = input("Enter the number of the sampling file you want to use: ")
try:
    index = int(choice) - 1
    if index < 0 or index >= len(sampling_files):
        raise IndexError
except (ValueError, IndexError):
    print("❌ Invalid selection.")
    exit(1)

chosen_file = os.path.join(sampling_dir, sampling_files[index])
print(f"✅ You selected: {sampling_files[index]}")

# --------- Load selected sample IDs ---------
with open(chosen_file, 'r') as f:
    selected_sample_ids = [line.strip() for line in f.readlines()]
print(f"Loaded {len(selected_sample_ids)} sample IDs from {sampling_files[index]}")

# --------- Load biome labels ---------
biome_labels = load_biome_labels(biome_labels_path)

# --------- Load filtered embeddings ---------
print("\nLoading filtered embeddings...")
start = time.time()
df_subbiomes, X_subbiomes = load_embeddings_h5(subbiomes_path, biome_labels, selected_sample_ids)
elapsed1 = time.time() - start
print(f"✅ Sub-biomes embeddings loaded in {elapsed1:.2f} seconds.")

start = time.time()
df_keywords, X_keywords = load_embeddings_h5(keywords_path, biome_labels, selected_sample_ids)
elapsed2 = time.time() - start
print(f"✅ Keywords embeddings loaded in {elapsed2:.2f} seconds.")

start = time.time()
df_sb_keywords, X_sb_keywords = load_embeddings_h5(sb_keywords_path, biome_labels, selected_sample_ids)
elapsed3 = time.time() - start
print(f"✅ Sub-biomes + keywords embeddings loaded in {elapsed3:.2f} seconds.")

print("\n✅ All embeddings loaded and ready for clustering.")







###################


# clustering: 




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
    print(f"✅ Saved plot to {out_path}")
    return fig

# --------- Helper: Filter out NaN rows ---------
def filter_nan_rows(X, df, label):
    nan_mask = np.isnan(X).any(axis=1)
    num_nans = np.sum(nan_mask)
    if num_nans > 0:
        print(f"⚠ Skipping {num_nans} samples due to NaNs in {label} embeddings")
    X_clean = X[~nan_mask]
    df_clean = df.loc[~nan_mask].reset_index(drop=True)
    return X_clean, df_clean

# --------- Helper: Print per-biome counts ---------
def print_biome_counts(df, label):
    print(f"\nSample counts per biome after filtering for {label}:")
    biome_counts = df['biome_label'].value_counts()
    for biome, count in biome_counts.items():
        print(f"  {biome}: {count}")
    print(f"  Total: {len(df)} samples")

# --------- Define biome colors ---------
biome_colors = {
    'water': '#8CC8CF',
    'plant': '#C0D184',
    'animal': '#C67D7B',
    'other': '#CCCCCC',
    'soil': '#CBBF82'
}

# --------- Filter NaNs before UMAP ---------
X_subbiomes_clean, df_subbiomes_clean = filter_nan_rows(X_subbiomes, df_subbiomes, 'sub-biomes')
X_keywords_clean, df_keywords_clean = filter_nan_rows(X_keywords, df_keywords, 'keywords')
X_sb_keywords_clean, df_sb_keywords_clean = filter_nan_rows(X_sb_keywords, df_sb_keywords, 'sub-biomes + keywords')

# --------- Print biome balance ---------
print_biome_counts(df_subbiomes_clean, 'sub-biomes')
print_biome_counts(df_keywords_clean, 'keywords')
print_biome_counts(df_sb_keywords_clean, 'sub-biomes + keywords')

# --------- Run UMAP reductions ---------
total_start = time.time()

start = time.time()
print("\nRunning UMAP on sub-biomes...")
df_subbiomes_clean = run_umap(X_subbiomes_clean, df_subbiomes_clean)
print(f"✅ Sub-biomes UMAP done in {time.time() - start:.2f} seconds.")

start = time.time()
print("Running UMAP on keywords...")
df_keywords_clean = run_umap(X_keywords_clean, df_keywords_clean)
print(f"✅ Keywords UMAP done in {time.time() - start:.2f} seconds.")

start = time.time()
print("Running UMAP on sub-biomes + keywords...")
df_sb_keywords_clean = run_umap(X_sb_keywords_clean, df_sb_keywords_clean)
print(f"✅ Sub-biomes + keywords UMAP done in {time.time() - start:.2f} seconds.")


# --------- Extract nspb and seed from sampling file name ---------
sampling_basename = os.path.splitext(sampling_files[index])[0]  # e.g., sampling_n5000_seed22
suffix = sampling_basename.replace('sampling_', '')  # e.g., n5000_seed22
suffix = suffix.replace('seed', 'rs')  


# --------- Define output paths ---------
out1 = os.path.join(work_dir, f'{suffix}_umap_subbiomes.html')
out2 = os.path.join(work_dir, f'{suffix}_umap_keywords.html')
out3 = os.path.join(work_dir, f'{suffix}_umap_sb_keywords.html')


# --------- Create and save plots ---------
start = time.time()
fig1 = make_scatter_plot(
    df_subbiomes_clean,
    title="Sub-biome embeddings",
    hover_fields={'sample_id': True, 'sub-biome': True, 'UMAP1': False, 'UMAP2': False},
    biome_colors=biome_colors,
    out_path=out1
)
print(f"✅ Sub-biome plot saved in {time.time() - start:.2f} seconds.")

start = time.time()
fig2 = make_scatter_plot(
    df_keywords_clean,
    title="Keyword embeddings",
    hover_fields={'sample_id': True, 'keywords': True, 'UMAP1': False, 'UMAP2': False},
    biome_colors=biome_colors,
    out_path=out2
)
print(f"✅ Keyword plot saved in {time.time() - start:.2f} seconds.")

start = time.time()
fig3 = make_scatter_plot(
    df_sb_keywords_clean,
    title="Sub-biomes + keyword (avg) embeddings",
    hover_fields={'sample_id': True, 'sub-biome': True, 'keywords': True, 'UMAP1': False, 'UMAP2': False},
    biome_colors=biome_colors,
    out_path=out3
)
print(f"✅ Sub-biomes + keywords plot saved in {time.time() - start:.2f} seconds.")

# --------- Optional: Show plots interactively ---------
fig1.show()
fig2.show()
fig3.show()

# --------- Final summary ---------
total_elapsed = time.time() - total_start
print(f"\n✅ Total clustering and plotting time: {total_elapsed/60:.2f} minutes")






from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# --------- Helper: Find optimal number of clusters using silhouette score ---------
def find_optimal_k(X, min_k=5, max_k=100, step=5):
    best_k = min_k
    best_score = -1
    scores = []

    print("\nEstimating optimal number of clusters using silhouette score...")
    for k in range(min_k, min(max_k, X.shape[0]), step):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) > 1:  # silhouette requires at least 2 clusters
            score = silhouette_score(X, labels)
            print(f"  k={k}: silhouette score = {score:.3f}")
            scores.append((k, score))
            if score > best_score:
                best_score = score
                best_k = k

    if not scores:
        raise ValueError("Failed to compute silhouette score for any k")

    # Optional: Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot([k for k, _ in scores], [s for _, s in scores], marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()

    print(f"✅ Best k determined: {best_k} with silhouette score {best_score:.3f}")
    return best_k

# --------- Helper: Cluster with best k and plot ---------
def cluster_and_plot(X, df, umap_x='UMAP1', umap_y='UMAP2', label_column='sub-biome', out_path=None):
    # Normalize embeddings
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Find optimal k
    best_k = find_optimal_k(X_normalized, min_k=5, max_k=100, step=5)

    # Run final clustering
    kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=1000)
    cluster_labels = kmeans.fit_predict(X_normalized)
    df['cluster_id'] = cluster_labels

    # Plot using plotly
    fig = px.scatter(
        df,
        x=umap_x, y=umap_y,
        color='cluster_id',
        color_continuous_scale='Viridis',
        hover_data={'sample_id': True, label_column: True, 'cluster_id': True},
        title=f"{label_column.capitalize()} clustered into {best_k} clusters (auto-detected)",
        height=800,
        width=1000
    )

    if out_path:
        fig.write_html(out_path)
        print(f"✅ Clustered plot saved to {out_path}")

    fig.show()
    return df, kmeans

# --------- Usage example with your sub-biomes ---------
df_subbiomes_clean, kmeans_model = cluster_and_plot(
    X_subbiomes_clean,
    df_subbiomes_clean,
    out_path=os.path.join(work_dir, f'{suffix}_subbiomes_clusters_auto.html')
)















import os
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# --------- Settings ---------
embeddings_h5_path = os.path.expanduser("~/cloudstor/Gaio/MicrobeAtlasProject/Hackathon/embeddings/GPT_sub_biomes_embeddings.h5")
output_csv = "sub_biome_to_cluster_kmeans1000.csv"
n_clusters = 1000

# --------- Load embeddings and texts ---------
print(f"🔹 Opening embeddings file: {embeddings_h5_path}")
with h5py.File(embeddings_h5_path, 'r') as f:
    print(f"Available datasets: {list(f.keys())}")
    
    embeddings = f['embeddings'][:]
    sample_ids = f['sample_ids'][:]
    
    if 'sub_texts' in f:
        texts = f['sub_texts'][:]
    elif 'texts' in f:
        texts = f['texts'][:]
    else:
        raise KeyError("Neither 'sub_texts' nor 'texts' found in the H5 file.")

# Decode text fields
print("🔹 Decoding texts and sample IDs...")
texts = np.array([txt.decode('utf-8') if isinstance(txt, bytes) else txt for txt in texts])
sample_ids = np.array([sid.decode('utf-8') if isinstance(sid, bytes) else sid for sid in sample_ids])

print(f"✅ Loaded {len(texts)} embeddings and texts.")






# --------- Run clustering ---------

from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

n_clusters = 1000
batch_size = 10000

print(f"🔹 Running MiniBatchKMeans (manual loop with progress) on {embeddings.shape[0]} embeddings...")

kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=batch_size,
    random_state=42,
    init='k-means++',
    max_iter=1  # We will manually handle epochs
)

# Initialize
kmeans.partial_fit(embeddings[:batch_size])

# Loop through batches
n_batches = int(np.ceil(embeddings.shape[0] / batch_size))

for epoch in range(10):  # number of epochs over data
    print(f"Epoch {epoch+1}/10")
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, embeddings.shape[0])
        batch = embeddings[start:end]
        kmeans.partial_fit(batch)

# Predict all
cluster_labels = kmeans.predict(embeddings)

print("✅ MiniBatchKMeans clustering done.")




# --------- Create mapping DataFrame ---------
print("🔹 Creating mapping DataFrame...")
df = pd.DataFrame({
    'sample_id': sample_ids,
    'sub_biome': texts,
    'cluster': cluster_labels
})

# --------- Optional: save only unique sub-biome to cluster ---------
df_unique = df[['sub_biome', 'cluster']].drop_duplicates()

# --------- Save ---------
df_unique.to_csv(output_csv, index=False)
print(f"✅ Saved mapping to {output_csv}")





