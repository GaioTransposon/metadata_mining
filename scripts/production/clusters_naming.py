#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:06:04 2025

@author: dgaio
"""



import os
import re
import h5py
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------
# Config (adjust as needed)
# ----------------------------
# Config
CLASS_A_NAME = "soil"
CLASS_B_NAME = "plant"
CLASS_A_TERMS = ["soil"]
CLASS_B_TERMS = ["plant"]
PER_CLASS = 1000
SEED = 42             # how many samples per group (soil/plant)
BASE_DIR = os.path.join(os.path.expanduser('~'), "MicrobeAtlasProject/Hackathon")
WORK_DIR = os.path.join(BASE_DIR, "embeddings")

# Text file mapping sample_id -> sub-biome text (one per line, tab-separated)
SUB_BIOMES_TXT = os.path.join(BASE_DIR, "GPT_sub_biomes.txt")   # e.g., "SRS123\tmaize rhizosphere soil"

# HDF5 with embeddings aligned to sample_ids
SUBBIOMES_H5  = os.path.join(WORK_DIR, "GPT_sub_biomes_embeddings_aligned.h5")
# If you prefer using a different H5 (e.g., keywords or combined), change the path above.

# ----------------------------
# Helpers
# ----------------------------
def load_subbiome_map(path: str) -> Dict[str, str]:
    """
    Load a tab-separated file mapping sample_id -> sub-biome string.
    - Skips lines with empty sub-biome text.
    - Prints how many entries were skipped.
    Returns dict {sample_id: sub_biome_text}.
    """
    mapping = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            sid, text = parts
            if not sid:  # skip empty IDs
                continue
            if not text:  # skip empty sub-biome
                skipped += 1
                continue
            mapping[sid] = text

    if skipped > 0:
        print(f"⚠️ Skipped {skipped:,} entries with empty sub-biome text.")
    print(f"✅ Loaded {len(mapping):,} entries into dictionary.")
    return mapping





def select_balanced_samples(
    id_to_text: Dict[str, str],
    class_a_name: str,
    class_b_name: str,
    class_a_terms: List[str],
    class_b_terms: List[str],
    per_class: int,
    seed: int,
    match_whole_words: bool = True,
    case_insensitive: bool = True,
    exclude_ambiguous: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a balanced subset between two classes using keyword matches.

    Parameters
    ----------
    id_to_text : {sample_id: text}
    class_a_name, class_b_name : display names for the two classes (e.g., "soil", "plant")
    class_a_terms, class_b_terms : keywords for each class (list of strings)
    per_class : target number of samples to pick for each class
    seed : RNG seed for reproducible shuffling
    match_whole_words : if True, wrap each term with word boundaries (\\b)
    case_insensitive : if True, use re.IGNORECASE
    exclude_ambiguous : if True, discard samples matching both classes

    Returns
    -------
    df_meta : DataFrame with columns [sample_id, sub_biome, group]
    selected_ids : list of sample_ids in the same order as df_meta
    """

    def _compile_pattern(terms: List[str]) -> re.Pattern:
        if match_whole_words:
            parts = [rf"\b{re.escape(t)}\b" for t in terms]
        else:
            parts = [re.escape(t) for t in terms]
        flags = re.IGNORECASE if case_insensitive else 0
        return re.compile("|".join(parts), flags=flags) if parts else re.compile(r"$^")  # matches nothing if empty

    pat_a = _compile_pattern(class_a_terms)
    pat_b = _compile_pattern(class_b_terms)

    rng = random.Random(seed)
    a_ids, b_ids = [], []

    for sid, text in id_to_text.items():
        t = text if isinstance(text, str) else str(text)
        has_a = bool(pat_a.search(t))
        has_b = bool(pat_b.search(t))

        if exclude_ambiguous and has_a and has_b:
            continue
        if has_a and not has_b:
            a_ids.append(sid)
        elif has_b and not has_a:
            b_ids.append(sid)
        elif not exclude_ambiguous and (has_a or has_b):
            # If we allow ambiguity, assign by priority: A first, else B
            a_ids.append(sid) if has_a else b_ids.append(sid)

    # Shuffle & pick
    rng.shuffle(a_ids)
    rng.shuffle(b_ids)
    a_pick = a_ids[:per_class]
    b_pick = b_ids[:per_class]

    # Warnings if underfilled
    if len(a_pick) < per_class or len(b_pick) < per_class:
        print(f"⚠️  Requested per_class={per_class}, but found "
              f"{class_a_name}={len(a_ids)} eligible, {class_b_name}={len(b_ids)} eligible "
              f"(after exclude_ambiguous={exclude_ambiguous}). Using the available minimum.")

    # Build metadata
    rows = (
        [{"sample_id": sid, "sub_biome": id_to_text[sid], "group": class_a_name} for sid in a_pick] +
        [{"sample_id": sid, "sub_biome": id_to_text[sid], "group": class_b_name} for sid in b_pick]
    )
    df_meta = pd.DataFrame(rows)
    selected_ids = df_meta["sample_id"].tolist()
    return df_meta, selected_ids












def load_embeddings_for_ids(h5_path: str, wanted_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings for a specific ordered list of sample IDs from an HDF5 file that has:
      - dataset: 'sample_ids'  (array of bytes/str)
      - dataset: 'embeddings'  (2D float array)
    Returns:
      X: embeddings aligned to wanted_ids (len = len(wanted_ids))
      found_ids: the subset that was actually found in the H5 (same order as X)
    Notes:
      - If some IDs are missing from the H5, they are skipped (and you’ll see a warning count).
    """
    wanted_set = set(wanted_ids)
    with h5py.File(h5_path, "r") as f:
        sid_raw = f["sample_ids"][:]
        # decode bytes to str if needed
        sids = np.array([s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in sid_raw])
        # map id -> index
        index = {sid: i for i, sid in enumerate(sids) if sid in wanted_set}
        found_ids = [sid for sid in wanted_ids if sid in index]
        if len(found_ids) < len(wanted_ids):
            missing = len(wanted_ids) - len(found_ids)
            print(f"⚠️  {missing} IDs not found in {os.path.basename(h5_path)} and will be skipped.")

        emb = f["embeddings"]
        X = np.vstack([emb[index[sid], :] for sid in found_ids]).astype(np.float32)
    return X, found_ids

# ----------------------------
# Main flow
# ----------------------------
if __name__ == "__main__":
    print("Loading sub-biome map...")
    id_to_text = load_subbiome_map(SUB_BIOMES_TXT)
    print(f"Loaded {len(id_to_text):,} entries from {os.path.basename(SUB_BIOMES_TXT)}.")

    print(f"Selecting balanced subset: {PER_CLASS} soil + {PER_CLASS} plant...")
    df_meta, selected_ids = select_balanced_samples(id_to_text, per_class=PER_CLASS, seed=SEED)
    print(f"Selected {len(df_meta):,} samples "
          f"({(df_meta['group']=='soil').sum()} soil, {(df_meta['group']=='plant').sum()} plant).")

    print("Loading embeddings for selected IDs...")
    X, found_ids = load_embeddings_for_ids(SUBBIOMES_H5, selected_ids)
    print(f"Embeddings loaded for {X.shape[0]:,} samples with dim={X.shape[1]}.")

    # Align df_meta to found_ids (in exact embedding order)
    df_meta = df_meta.set_index("sample_id").loc[found_ids].reset_index()
    df_meta.rename(columns={"index": "sample_id"}, inplace=True)

    # Quick sanity checks & summaries
    print("\nPreview of metadata:")
    print(df_meta.head(10).to_string(index=False))
    print("\nGroup counts (after alignment to found embeddings):")
    print(df_meta["group"].value_counts())

    # At this point, you have:
    # - df_meta: columns = [sample_id, sub_biome, group]
    # - X:      np.ndarray shape (N_found, 1536) or whatever your embedding dim is
    # You can proceed to PCA, graph building, etc., *only* on this balanced subset.








# pca_leiden_subset.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

import igraph as ig
import leidenalg


# -------------------------------------------------
# Inputs expected from the previous step:
#   - df_meta: DataFrame with columns [sample_id, sub_biome, group]
#   - X:       np.ndarray of shape (N, D) embeddings aligned to df_meta
# If you saved them, load here. Otherwise, assume they are in memory.
# -------------------------------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (cosine-friendly)."""
    return normalize(X, norm='l2', axis=1)

def run_pca(X: np.ndarray, n_components: int = 256, random_state: int = 42) -> tuple[PCA, np.ndarray]:
    """
    PCA on normalized embeddings. Returns (pca_model, X_pca).
    For ~2k samples this is fast; for millions use IncrementalPCA later.
    """
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def build_knn_graph(
    X: np.ndarray, k: int = 30, metric: str = "cosine"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build kNN using sklearn NearestNeighbors.
    Returns (neighbors, sims) where:
      neighbors: (N, k) int indices of neighbors
      sims:      (N, k) cosine similarities to those neighbors
    Note: sklearn 'cosine' distance = 1 - cosine_similarity.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    dists, neigh = nn.kneighbors(X, return_distance=True)
    # drop self at column 0
    neigh = neigh[:, 1:]
    dists = dists[:, 1:]
    sims = 1.0 - dists  # cosine similarity = 1 - cosine distance
    return neigh, sims

def leiden_from_knn(neigh: np.ndarray, sims: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """
    Create an undirected weighted graph from kNN and run Leiden.
    We symmetrize by adding edges both (i->j) and (j->i) via max weight.
    """
    n = neigh.shape[0]
    # Build an edge map to keep the max weight when edges repeat (from symmetric insertions)
    edge_map = {}  # (min(i,j), max(i,j)) -> weight
    for i in range(n):
        for j, w in zip(neigh[i], sims[i]):
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a == b:
                continue
            key = (a, b)
            if key not in edge_map or w > edge_map[key]:
                edge_map[key] = float(w)

    # Convert to igraph
    edges = list(edge_map.keys())
    weights = list(edge_map.values())
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )
    return np.array(part.membership, dtype=np.int32)


def summarize_clustering(df_meta: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Prints:
      - cluster sizes,
      - ARI/NMI (label-invariant),
      - raw crosstab of true group vs cluster (strings vs ints OK),
      - aggregated 2-class confusion by mapping each cluster -> its majority class.
    """
    df = df_meta.copy()
    df["cluster"] = labels

    print("\nCluster sizes (top 10):")
    print(df["cluster"].value_counts().head(10))

    # ARI/NMI (label-invariant): map true groups to ints; predicted are cluster ints already
    groups = sorted(df["group"].unique())
    g2i = {g: i for i, g in enumerate(groups)}
    y_true_int = df["group"].map(g2i).to_numpy()
    y_pred_int = df["cluster"].to_numpy()

    ari = adjusted_rand_score(y_true_int, y_pred_int)
    nmi = normalized_mutual_info_score(y_true_int, y_pred_int)
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")

    # Raw crosstab (no type conflict)
    xtab = pd.crosstab(df["group"], df["cluster"])
    print("\nCrosstab (rows=true group, cols=cluster):")
    print(xtab)

    # Majority-vote aggregation: cluster -> predicted group
    majority_map = {}
    for c, sub in df.groupby("cluster"):
        majority = sub["group"].value_counts().idxmax()
        majority_map[c] = majority

    df["pred_group"] = df["cluster"].map(majority_map)

    agg_xtab = pd.crosstab(df["group"], df["pred_group"])
    acc = np.trace(agg_xtab.reindex(index=groups, columns=groups, fill_value=0).to_numpy()) / len(df)
    print("\nAggregated 2-class confusion (after majority-vote per cluster):")
    print(agg_xtab)
    print(f"\nAggregated accuracy (majority mapping): {acc:.4f}")

    # (Optional) show which clusters lean soil vs plant
    leaning = (
        df.groupby("cluster")["group"]
          .value_counts(normalize=True)
          .rename("ratio")
          .reset_index()
          .sort_values(["cluster", "ratio"], ascending=[True, False])
    )
    print("\nTop class per cluster (first 10 clusters):")
    print(leaning.groupby("cluster").head(1).head(10))


# -------------------------------------------------
# NEW: Helper functions you asked for
# -------------------------------------------------

def Xp_builder(X: np.ndarray, pca_components: int = 256, normalize_first: bool = True) -> tuple[PCA, np.ndarray]:
    """
    Build the feature matrix for graph building (usually PCA of normalized embeddings).
    Returns (pca_model, Xp).
    """
    X_in = l2_normalize(X) if normalize_first else X
    pca, Xp = run_pca(X_in, n_components=min(pca_components, X_in.shape[1]))
    return pca, Xp

def leiden_runner(Xp: np.ndarray, k: int, resolution: float) -> np.ndarray:
    """
    Given a feature matrix Xp (e.g., PCA output), build a cosine kNN graph and run Leiden.
    Returns the cluster labels (np.ndarray, shape (N,)).
    """
    neigh, sims = build_knn_graph(Xp, k=k, metric="cosine")
    labels = leiden_from_knn(neigh, sims, resolution=resolution)
    return labels

def small_grid_search(
    df_meta: pd.DataFrame,
    X: np.ndarray,
    pca_components_list=(128, 192, 256),
    k_list=(15, 30, 50),
    resolution_list=(0.6, 0.8, 1.0, 1.2),
) -> pd.DataFrame:
    """
    Sweep a small grid of (pca_components, k, resolution).
    Prints quick results and returns a DataFrame sorted by ARI then NMI.
    """
    results = []
    groups = sorted(df_meta["group"].unique())
    g2i = {g: i for i, g in enumerate(groups)}
    y_true_int = df_meta["group"].map(g2i).to_numpy()

    for pcomps in pca_components_list:
        pca, Xp = Xp_builder(X, pca_components=pcomps, normalize_first=True)
        for k in k_list:
            for res in resolution_list:
                labels = leiden_runner(Xp, k=k, resolution=res)
                ari = adjusted_rand_score(y_true_int, labels)
                nmi = normalized_mutual_info_score(y_true_int, labels)
                n_clusters = len(np.unique(labels))
                print(f"pca={pcomps:<3} k={k:<2} res={res:<3} -> clusters={n_clusters:<3}  ARI={ari:.4f}  NMI={nmi:.4f}")
                results.append({
                    "pca_components": pcomps,
                    "k": k,
                    "resolution": res,
                    "n_clusters": n_clusters,
                    "ARI": ari,
                    "NMI": nmi,
                })

    res_df = pd.DataFrame(results).sort_values(["ARI", "NMI"], ascending=[False, False]).reset_index(drop=True)
    return res_df


def main_pipeline(
    df_meta: pd.DataFrame,
    X: np.ndarray,
    pca_components: int = 256,
    k: int = 30,
    resolution: float = 1.0,
):
    # 1) Normalize (cosine-friendly) + PCA
    pca, Xp = Xp_builder(X, pca_components=pca_components, normalize_first=True)

    print(f"PCA done. Explained variance ratio (first 10 comps):")
    print(np.round(pca.explained_variance_ratio_[:10], 4))
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # 2) kNN graph on PCA space (cosine) + Leiden
    neigh, sims = build_knn_graph(Xp, k=k, metric="cosine")
    print(f"kNN built with k={k}.")

    labels = leiden_from_knn(neigh, sims, resolution=resolution)
    n_clusters = len(np.unique(labels))
    print(f"Leiden found {n_clusters} clusters (resolution={resolution}).")

    # 3) Quick diagnostics
    summarize_clustering(df_meta, labels)

    # Return what we might need later
    return {
        "pca_model": pca,
        "Xp": Xp,
        "labels": labels,
        "neigh": neigh,
        "sims": sims,
        "params": {"k": k, "resolution": resolution, "pca_components": pca.n_components_},
    }

# -------------------------
# If running as a script:
# -------------------------
if __name__ == "__main__":
    # You can import df_meta, X from your previous step module,
    # or load from disk if you saved them. Here we assume they exist.
    try:
        df_meta
        X
    except NameError:
        raise RuntimeError(
            "df_meta and X must be defined in the current session. "
            "Run the selection/embedding-loading step first, or load them from disk."
        )

    results = main_pipeline(
        df_meta=df_meta,
        X=X,
        pca_components=256,   # try 128/192 too
        k=30,                 # try 15–50
        resolution=1.0        # try 0.6–1.4
    )

    # Example: attach cluster labels back to df_meta for later use
    df_meta["cluster"] = results["labels"]
    # Save if you want
    # df_meta.to_csv("soil_plant_leiden_labels.tsv", sep="\t", index=False)

    # OPTIONAL: quick sweep after the first run
    # grid = small_grid_search(
    #     df_meta=df_meta,
    #     X=X,
    #     pca_components_list=(128, 192, 256),
    #     k_list=(15, 30, 50),
    #     resolution_list=(0.6, 0.8, 1.0, 1.2),
    # )
    # print("\nGrid search summary (top 10 by ARI):")
    # print(grid.head(10))
    
    
















import re
import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# -----------------------------
# Text prep
# -----------------------------
DOMAIN_STOP = {
    "sample","samples","swab","host","environment","environmental","source",
    "female","male","human","plant","organism","specimen","biome","subbiome",
    "unknown","unspecified","na","misc","other",
    "zone","area","region","site","location",
    *list(ENGLISH_STOP_WORDS),
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[‘’ʼ´`]", "'", s)
    return s

def analyzer(text: str) -> List[str]:
    text = normalize_text(text)
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in DOMAIN_STOP and len(t) > 1]

# -----------------------------
# Vectorization
# -----------------------------
def build_vectorizer(min_df=5, max_df=0.9, ngram_range=(1,3)):
    return CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

def doc_term_matrices(texts: List[str], vectorizer: CountVectorizer):
    X = vectorizer.fit_transform(texts)      # csr (N_docs, V)
    vocab = np.array(vectorizer.get_feature_names_out())
    df = (X > 0).sum(axis=0).A1              # document frequency
    return X, vocab, df

# -----------------------------
# Scoring (no length bonus)
# -----------------------------
def cluster_phrase_scores(X, labels: np.ndarray, vocab: np.ndarray, alpha=1.5):
    """
    score = df_c * exclusivity^alpha
    where:
      df_c = document frequency of term within cluster
      exclusivity = P(term|cluster) / P(term)  (based on tf; smoothed)
    """
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    V = X.shape[1]

    # per-cluster indices
    idx_by_c = {c: np.where(labels == c)[0] for c in clusters}
    n_by_c = {c: len(idx) for c, idx in idx_by_c.items()}

    # Global
    tf_total = X.sum(axis=0).A1
    P_t = (tf_total + 1) / (tf_total.sum() + V)  # smoothed

    scores_by_c = {}
    details_by_c = {}

    for c, idx in idx_by_c.items():
        subX = X[idx]
        tf_c = subX.sum(axis=0).A1
        df_c = (subX > 0).sum(axis=0).A1

        # probabilities
        P_t_c = (tf_c + 1) / (tf_c.sum() + V)
        exclusivity = P_t_c / P_t

        score = df_c * np.power(exclusivity, alpha)

        scores_by_c[c] = score
        details_by_c[c] = {
            "df": df_c,
            "tf": tf_c,
            "exclusivity": exclusivity,
            "cluster_size": n_by_c[c]
        }

    return scores_by_c, details_by_c

# -----------------------------
# Term picking (by score)
# -----------------------------
def pick_term_indices_for_cluster(scores: np.ndarray, vocab: np.ndarray, top_k=3, min_chars=2) -> List[int]:
    idx_sorted = np.argsort(scores)[::-1]
    chosen, chosen_terms = [], []
    for j in idx_sorted:
        term = vocab[j]
        if len(term) < min_chars:
            continue
        # prevent trivial near-duplicates (substring containment)
        if any(term in t or t in term for t in chosen_terms):
            continue
        if scores[j] <= 0:
            continue
        chosen.append(j)
        chosen_terms.append(term)
        if len(chosen) >= top_k:
            break
    return chosen

# -----------------------------
# Name building with collision resolution
# -----------------------------
def build_unique_names_progressive(c2terms: Dict[int, List[str]],
                                   joiner: str = "_",
                                   keep_id_tag: bool = True) -> Dict[int, str]:
    """
    Try 1-term names; resolve collisions by adding 2nd term, then 3rd.
    If still colliding, append _c{id}.
    """
    # Stage 1: first term
    names1 = {c: (terms[0] if terms else f"cluster_{c}") for c, terms in c2terms.items()}
    inv1 = defaultdict(list)
    for c, n in names1.items():
        inv1[n].append(c)

    final = {}
    collisions = []

    for name, cs in inv1.items():
        if len(cs) == 1:
            final[cs[0]] = name
        else:
            collisions.extend(cs)

    # Stage 2: first_two
    if collisions:
        names2 = {}
        for c in collisions:
            terms = c2terms.get(c, [])
            if len(terms) >= 2:
                names2[c] = joiner.join(terms[:2])
            else:
                names2[c] = names1[c]
        inv2 = defaultdict(list)
        for c, n in names2.items():
            inv2[n].append(c)

        new_collisions = []
        for name, cs in inv2.items():
            if len(cs) == 1:
                final[cs[0]] = name
            else:
                new_collisions.extend(cs)
        collisions = new_collisions

    # Stage 3: first_three
    if collisions:
        names3 = {}
        for c in collisions:
            terms = c2terms.get(c, [])
            if len(terms) >= 3:
                names3[c] = joiner.join(terms[:3])
            else:
                names3[c] = (joiner.join(terms) if terms else f"cluster_{c}")
        inv3 = defaultdict(list)
        for c, n in names3.items():
            inv3[n].append(c)

        new_collisions = []
        for name, cs in inv3.items():
            if len(cs) == 1:
                final[cs[0]] = name
            else:
                new_collisions.extend(cs)
        collisions = new_collisions

    # Stage 4: still colliding → append _c{id}
    for c in collisions:
        base = names1.get(c)  # fall back to first term base (or use names3[c])
        if not base:
            base = f"cluster_{c}"
        final[c] = f"{base}_c{c}" if keep_id_tag else f"{base}_{c}"

    return final

# -----------------------------
# Formatting counts
# -----------------------------
def format_terms_with_counts(term_idx: List[int], vocab: np.ndarray, df_c: np.ndarray, cluster_size: int) -> str:
    parts = []
    for j in term_idx:
        term = vocab[j]
        cnt = int(df_c[j])
        pct = 100.0 * cnt / max(cluster_size, 1)
        parts.append(f"{term} ({cnt}, {pct:.0f}%)")
    return ", ".join(parts)

# -----------------------------
# Public API
# -----------------------------
def name_clusters_from_texts(
    df_meta: pd.DataFrame,
    labels: np.ndarray,
    text_col: str = "sub_biome",
    min_df: int = 5,
    max_df: float = 0.9,
    ngram_range=(1,3),
    alpha: float = 1.5,
    top_terms_per_cluster: int = 3,
    joiner: str = "_",
    keep_id_tag: bool = True
) -> pd.DataFrame:
    """
    Returns one row per cluster with human-readable names and counts.
    Columns: [cluster_id, size, name, alt_names, top_terms, top_terms_counts]
    """
    texts = df_meta[text_col].fillna("").astype(str).tolist()

    # 1) Vectorize
    vect = build_vectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X, vocab, _ = doc_term_matrices(texts, vect)

    # 2) Scores per cluster (abundance-first, no length bonus)
    scores_by_c, details_by_c = cluster_phrase_scores(X, labels, vocab, alpha=alpha)

    sizes = pd.Series(labels).value_counts().to_dict()
    c2term_indices, c2terms = {}, {}

    # 3) Choose top terms
    for c, scores in scores_by_c.items():
        idxs = pick_term_indices_for_cluster(scores, vocab, top_k=top_terms_per_cluster, min_chars=2)
        c2term_indices[c] = idxs
        c2terms[c] = [vocab[j] for j in idxs] if idxs else [f"cluster_{c}"]

    # 4) Build unique names with progressive concatenation
    unique_names = build_unique_names_progressive(c2terms, joiner=joiner, keep_id_tag=keep_id_tag)

    # 5) Assemble table (with counts)
    rows = []
    for c in sorted(scores_by_c.keys()):
        size = int(sizes.get(int(c), 0))
        df_c = details_by_c[c]["df"]
        idxs = c2term_indices[c]
        terms = [vocab[j] for j in idxs] if idxs else [f"cluster_{c}"]

        name = unique_names[c]
        # alt_names are the selected terms not in the display name (simple filter)
        alt = [t for t in terms if t not in set(name.split(joiner))]

        top_terms_counts = format_terms_with_counts(idxs, vocab, df_c, cluster_size=size) if idxs else ""

        rows.append({
            "cluster_id": int(c),
            "size": size,
            "name": name,
            "alt_names": ", ".join(alt),
            "top_terms": ", ".join(terms),
            "top_terms_counts": top_terms_counts
        })

    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)



named = name_clusters_from_texts(
    df_meta=df_meta,
    labels=results["labels"],
    text_col="sub_biome",
    min_df=5,
    max_df=0.9,
    ngram_range=(1,3),
    alpha=1.5,                 # exclusivity weight; lower → pure frequency, higher → more specific
    top_terms_per_cluster=3,
    joiner="_",                # <-- underscores per your preference
    keep_id_tag=True
)
print(named.head(15).to_string(index=False))







import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def pca_scatter_named(Xp, labels, named_df, title="PCA (2D) — clusters named"):
    """
    Xp: PCA output (N, d) from main_pipeline
    labels: cluster ids (N,)
    named_df: DataFrame from name_clusters_from_texts()
              with columns ['cluster_id','name']
    """
    if Xp.shape[1] < 2:
        raise ValueError("Xp needs at least 2 components for 2D scatter.")

    # build mapping cluster_id -> name
    id2name = dict(zip(named_df["cluster_id"], named_df["name"]))
    cluster_names = np.array([id2name.get(c, f"c{c}") for c in labels])

    x, y = Xp[:, 0], Xp[:, 1]
    plt.figure(figsize=(9, 7))

    # get unique cluster names in sorted order by cluster size
    sizes = named_df.set_index("name")["size"]
    unique_names = named_df.sort_values("size", ascending=False)["name"].tolist()

    cmap = get_cmap("tab20")
    for i, cname in enumerate(unique_names):
        mask = cluster_names == cname
        plt.scatter(
            x[mask], y[mask],
            s=12,
            alpha=0.7,
            c=[cmap(i % cmap.N)],
            label=f"{cname} (n={sizes[cname]})"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Clusters")
    plt.tight_layout()
    plt.show()



results = main_pipeline(df_meta, X, pca_components=256, k=30, resolution=1.0)
labels = results["labels"]
Xp = results["Xp"]

named = name_clusters_from_texts(df_meta, labels, text_col="sub_biome")

# visualize
pca_scatter_named(Xp, labels, named, title="PCA — clusters with names")
















