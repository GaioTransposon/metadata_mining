#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:29:25 2026

@author: dgaio
"""

#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
except ImportError:
    raise ImportError("Please install scikit-learn: conda install scikit-learn")


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def lenient_match(a, b):
    a = clean(a)
    b = clean(b)
    return bool(a and b and (a in b or b in a))


def read_two_col(path, value_col):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["sample", value_col],
        dtype=str,
        keep_default_na=False
    )


def load_gold_dict(path, biome_index=1, subbiome_index=0):
    with open(path, "rb") as f:
        gold = pickle.load(f)

    rows = []
    for sample, value in gold.items():
        if isinstance(value, (list, tuple)):
            biome = value[biome_index] if len(value) > biome_index else ""
            subbiome = value[subbiome_index] if len(value) > subbiome_index else ""
        elif isinstance(value, dict):
            biome = value.get("biome", "")
            subbiome = (
                value.get("sub-biome")
                or value.get("sub_biome")
                or value.get("subbiome")
                or ""
            )
        else:
            biome = ""
            subbiome = ""

        rows.append({
            "sample": str(sample),
            "gold_biome": biome,
            "gold_sub_biome": subbiome
        })

    return pd.DataFrame(rows)


def print_metrics(title, df, gold_col, pred_col):
    df = df.copy()
    df[gold_col] = df[gold_col].map(clean)
    df[pred_col] = df[pred_col].map(clean)
    df = df[(df[gold_col] != "") & (df[pred_col] != "")]

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Comparable samples: {len(df)}")

    if len(df) == 0:
        return

    exact = (df[gold_col] == df[pred_col]).mean()
    lenient = df.apply(lambda r: lenient_match(r[gold_col], r[pred_col]), axis=1).mean()

    print(f"Exact agreement:     {exact:.3f}")
    print(f"Lenient agreement:   {lenient:.3f}")
    print(f"Accuracy:            {accuracy_score(df[gold_col], df[pred_col]):.3f}")
    print(f"Macro precision:     {precision_score(df[gold_col], df[pred_col], average='macro', zero_division=0):.3f}")
    print(f"Macro recall:        {recall_score(df[gold_col], df[pred_col], average='macro', zero_division=0):.3f}")
    print(f"Macro F1:            {f1_score(df[gold_col], df[pred_col], average='macro', zero_division=0):.3f}")
    print(f"Cohen kappa:         {cohen_kappa_score(df[gold_col], df[pred_col]):.3f}")

    print("\nTop gold labels:")
    print(df[gold_col].value_counts().head(10).to_string())

    print("\nTop predicted labels:")
    print(df[pred_col].value_counts().head(10).to_string())


def load_h5(path):
    if h5py is None:
        raise ImportError("h5py is required for embedding comparison")

    with h5py.File(path, "r") as h5:
        ids = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in h5["sample_ids"][:]
        ]
        emb = np.asarray(h5["embeddings"][:], dtype=np.float32)

    return dict(zip(ids, emb))


def compare_embeddings(gpt_h5, gold_h5):
    if not os.path.exists(gpt_h5) or not os.path.exists(gold_h5):
        return

    gpt = load_h5(gpt_h5)
    gold = load_h5(gold_h5)

    common = sorted(set(gpt) & set(gold))

    print("\n" + "=" * 70)
    print("SUB-BIOME EMBEDDING SIMILARITY")
    print("=" * 70)
    print(f"Common embedded samples: {len(common)}")

    if not common:
        return

    A = np.vstack([gpt[s] for s in common])
    B = np.vstack([gold[s] for s in common])

    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)

    cosine = np.sum(A * B, axis=1)

    rng = np.random.default_rng(22)
    shuffled = rng.permutation(len(common))
    background = np.sum(A * B[shuffled], axis=1)

    print(f"Mean cosine GPT vs gold:       {cosine.mean():.3f}")
    print(f"Median cosine GPT vs gold:     {np.median(cosine):.3f}")
    print(f"SD cosine GPT vs gold:         {cosine.std():.3f}")
    print(f"5th percentile:                {np.percentile(cosine, 5):.3f}")
    print(f"95th percentile:               {np.percentile(cosine, 95):.3f}")
    print(f"Mean shuffled-background cos:  {background.mean():.3f}")
    print(f"Lift over background:          {cosine.mean() - background.mean():.3f}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--work_dir", required=True)
    p.add_argument("--gold_dict", default=None)
    p.add_argument("--gpt_biomes", default="GPT_biomes.txt")
    p.add_argument("--gpt_sub_biomes", default="GPT_sub_biomes.txt")
    p.add_argument("--gold_biome_index", type=int, default=1)
    p.add_argument("--gold_subbiome_index", type=int, default=0)
    p.add_argument("--embedding_dim", type=int, default=1536)

    args = p.parse_args()

    work_dir = os.path.expanduser(args.work_dir)

    gold_dict = (
        os.path.expanduser(args.gold_dict)
        if args.gold_dict
        else os.path.join(os.path.dirname(work_dir), "gold_dict.pkl")
    )

    gold = load_gold_dict(
        gold_dict,
        biome_index=args.gold_biome_index,
        subbiome_index=args.gold_subbiome_index
    )

    gpt_biomes = read_two_col(os.path.join(work_dir, args.gpt_biomes), "gpt_biome")
    gpt_sub = read_two_col(os.path.join(work_dir, args.gpt_sub_biomes), "gpt_sub_biome")

    biome_df = gold.merge(gpt_biomes, on="sample", how="inner")
    sub_df = gold.merge(gpt_sub, on="sample", how="inner")

    print_metrics(
        "BIOME PERFORMANCE: GPT vs ground truth",
        biome_df,
        "gold_biome",
        "gpt_biome"
    )

    print_metrics(
        "SUB-BIOME TEXT PERFORMANCE: GPT vs ground truth",
        sub_df,
        "gold_sub_biome",
        "gpt_sub_biome"
    )

    gpt_h5 = os.path.join(
        work_dir,
        f"embeddings/GPT_sub_biomes_embeddings_{args.embedding_dim}.h5"
    )
    gold_h5 = os.path.join(
        work_dir,
        f"embeddings/gold_sub_biomes_embeddings_{args.embedding_dim}.h5"
    )

    compare_embeddings(gpt_h5, gold_h5)


if __name__ == "__main__":
    main()